// src/net/stage_tls.cpp - see include/strata/net/stage_tls.hpp.
#if defined(_WIN32)
// OpenSSL's headers bring in windows.h: without its min/max macros, and without wincrypt.h's X509_NAME
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "strata/net/stage_tls.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <set>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#if defined(_WIN32)
#include <windows.h>   // (before OpenSSL's headers, which undo what wincrypt.h may define)
#include <aclapi.h>
#include <io.h>
#include <sddl.h>
#else
#include <unistd.h>
#endif

#if defined(STRATA_HAVE_OPENSSL)
#include <openssl/bio.h>
#include <openssl/bn.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/opensslv.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/ssl.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#if OPENSSL_VERSION_NUMBER < 0x10101000L
#error "the remote stage's TLS 1.3 needs OpenSSL 1.1.1 or later"
#endif
#endif

namespace strata::net {

namespace fs = std::filesystem;

std::string stage_tls_dir() {
    const char* v = std::getenv("STRATA_STAGE_TLS");
    return v != nullptr ? v : "";
}

std::string stage_host_name() {
    std::string s;
#if defined(_WIN32)
    const char* h = std::getenv("COMPUTERNAME");
    if (h != nullptr) s = h;
#else
    char h[256] = {};
    if (gethostname(h, sizeof h - 1) == 0) s = h;   // (a shell's HOSTNAME is often not exported)
#endif
    return s.empty() ? "strata" : s;
}

std::string stage_tls_short(const std::string& fp) { return fp.size() > 23 ? fp.substr(0, 23) : fp; }

namespace {

std::atomic<uint64_t> g_rekey_bytes{0}, g_key_updates{0};
constexpr uint64_t kRekeyBytes = (uint64_t) 16 << 30;   // 2^20 full records, against TLS 1.3's 2^24.5 for AES-GCM

}  // namespace

void stage_tls_set_rekey_bytes(uint64_t n) { g_rekey_bytes = n; }
uint64_t stage_tls_key_updates() { return g_key_updates.load(); }

#if defined(STRATA_HAVE_OPENSSL)

namespace {

// OpenSSL without this PC's openssl.cnf: the link sets its own versions, suites, groups and certificates, so a host's
// configuration (a distribution's crypto policy, or one newer than a release's static OpenSSL reads) changes nothing.
// The first init decides; later implicit ones do not load it either.
void init_openssl() {
    static const bool done = OPENSSL_init_ssl(OPENSSL_INIT_NO_LOAD_CONFIG, nullptr) == 1;
    (void) done;
}

// a path the engine got as UTF-8 (as every path it reads: #3202)
fs::path u8path(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }
std::string path_text(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

#if defined(_WIN32)
// what a handle opened with FILE_FLAG_OPEN_REPARSE_POINT is: a link (a symbolic link, or a junction, which any user
// may make and std::filesystem does not call a symlink), and a folder
bool what_is(HANDLE h, bool& link, bool& folder) {
    FILE_ATTRIBUTE_TAG_INFO ti {};
    if (!GetFileInformationByHandleEx(h, FileAttributeTagInfo, &ti, sizeof ti)) return false;
    link = (ti.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 && IsReparseTagNameSurrogate(ti.ReparseTag);
    folder = (ti.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    return true;
}
HANDLE open_itself(const fs::path& p, DWORD access,
                   DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE) {
    return CreateFileW(p.c_str(), access, share, nullptr, OPEN_EXISTING,
                       FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
}
#endif

// the key folder held open while the keygen writes into it: on Windows not shared for delete, so that nobody moves it
// away and puts a link in its place meanwhile (POSIX: a folder of this user's, mode 0700, that no other user may
// rename under a sticky parent); nothing to hold elsewhere
struct Hold {
#if defined(_WIN32)
    HANDLE h = INVALID_HANDLE_VALUE;
    ~Hold() {
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
    }
#endif
    Hold() = default;
    Hold(const Hold&) = delete;
    Hold& operator=(const Hold&) = delete;
};

// whether p is a link (on Windows a junction too); false when it is not there
bool is_link(const fs::path& p) {
    std::error_code ec;
    if (fs::is_symlink(p, ec)) return true;
#if defined(_WIN32)
    const HANDLE h = open_itself(p, FILE_READ_ATTRIBUTES);
    if (h == INVALID_HANDLE_VALUE) return false;
    bool link = false, folder = false;
    const bool known = what_is(h, link, folder);
    CloseHandle(h);
    return known && link;
#else
    return false;
#endif
}

// The key folder for this user alone, as OpenSSH asks of a key's: on Linux mode 0700; on Windows a protected DACL
// (nothing inherited: C:\ lets every signed-in user change what is made under it) giving this user, SYSTEM and
// Administrators full control, inherited by what the folder holds and will hold. A folder another user owns is
// refused, its permissions not changed (an owner may change them again whatever they say; on Windows the owner may be
// the Administrators, as an elevated prompt makes what it makes), and so is a link: the folder itself is opened once
// (not what a link names) and looked at and changed through that handle, so a folder swapped in meanwhile is not
// what is changed; on Windows that handle is kept in `hold`
bool owner_only(const fs::path& p, std::string& err, Hold& hold) {
#if defined(_WIN32)
    std::vector<unsigned char> user;
    HANDLE tok = nullptr;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &tok)) {
        DWORD n = 0;
        (void) GetTokenInformation(tok, TokenUser, nullptr, 0, &n);
        user.resize(n);
        if (n == 0 || !GetTokenInformation(tok, TokenUser, user.data(), n, &n)) user.clear();
        CloseHandle(tok);
    }
    const PSID me = user.empty() ? nullptr : ((TOKEN_USER*) user.data())->User.Sid;
    std::string sddl;
    LPSTR sid = nullptr;
    if (me != nullptr && ConvertSidToStringSidA(me, &sid)) {
        sddl = std::string("D:P(A;OICI;FA;;;") + sid + ")(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)";
        LocalFree(sid);
    }
    // not shared for delete (FILE_LIST_DIRECTORY: a share mode counts only with a data access): a process that holds
    // the folder with the right to delete or move it - one that opened it while C:\'s inheritance let every user do
    // so - makes this open fail, and none may open it so while the keygen holds it
    HANDLE h = open_itself(p, READ_CONTROL | WRITE_DAC | FILE_READ_ATTRIBUTES | FILE_LIST_DIRECTORY,
                           FILE_SHARE_READ | FILE_SHARE_WRITE);
    const DWORD oe = h == INVALID_HANDLE_VALUE ? GetLastError() : ERROR_SUCCESS;
    if (oe == ERROR_SHARING_VIOLATION) {
        err = path_text(p) + " is held open by another process (it may move or delete the folder): close it, or give "
              "a new folder";
        return false;
    }
    const bool may_change = h != INVALID_HANDLE_VALUE;
    if (!may_change) h = open_itself(p, READ_CONTROL | FILE_READ_ATTRIBUTES);   // (to say why: another user's, ...)
    if (h == INVALID_HANDLE_VALUE) {
        err = "cannot open " + path_text(p) + " (Windows error " + std::to_string(oe) + ")";
        return false;
    }
    std::string why;
    DWORD rc = ERROR_SUCCESS;
    bool link = false, folder = false;
    PSID owner = nullptr;
    PSECURITY_DESCRIPTOR osd = nullptr;
    if (!what_is(h, link, folder))
        rc = GetLastError();
    else if (link)
        why = " is a link: give the folder itself";
    else if (!folder)
        why = " is not a folder";
    else if ((rc = GetSecurityInfo(h, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION, &owner, nullptr, nullptr, nullptr,
                                   &osd)) == ERROR_SUCCESS &&
             !(me != nullptr && EqualSid(owner, me)) && !IsWellKnownSid(owner, WinBuiltinAdministratorsSid))
        why = " is another user's folder: the key needs one of this user's";
    if (osd != nullptr) LocalFree(osd);
    if (rc == ERROR_SUCCESS && why.empty()) {
        PSECURITY_DESCRIPTOR sd = nullptr;
        PACL dacl = nullptr;
        BOOL present = FALSE, defaulted = FALSE;
        rc = !may_change ? ERROR_ACCESS_DENIED : ERROR_INVALID_SID;
        if (may_change && !sddl.empty() &&
            ConvertStringSecurityDescriptorToSecurityDescriptorA(sddl.c_str(), SDDL_REVISION_1, &sd, nullptr) &&
            GetSecurityDescriptorDacl(sd, &present, &dacl, &defaulted))
            rc = SetSecurityInfo(h, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                                 nullptr, nullptr, dacl, nullptr);
        if (sd != nullptr) LocalFree(sd);
    }
    if (!why.empty() || rc != ERROR_SUCCESS) {
        CloseHandle(h);
        err = !why.empty() ? path_text(p) + why
                           : "cannot make " + path_text(p) + " this user's alone (Windows error " + std::to_string(rc) +
                                 ")";
        return false;
    }
    hold.h = h;
#else
    (void) hold;
    const int fd = ::open(p.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        const int e = errno;   // (a link: ELOOP, or on Linux ENOTDIR, O_DIRECTORY being looked at first)
        struct stat ls {};
        const bool link = e == ELOOP || (e == ENOTDIR && ::lstat(p.c_str(), &ls) == 0 && S_ISLNK(ls.st_mode));
        err = link ? path_text(p) + " is a link: give the folder itself"
                   : "cannot open " + path_text(p) + ": " + std::strerror(e);
        return false;
    }
    std::string why;
    struct stat st {};
    if (::fstat(fd, &st) != 0)
        why = "cannot read " + path_text(p) + ": " + std::strerror(errno);
    else if (st.st_uid != ::geteuid())   // (root may change another user's folder: refused before it does)
        why = path_text(p) + " is another user's folder: the key needs one of this user's";
    else if (::fchmod(fd, S_IRWXU) != 0)
        why = "cannot make " + path_text(p) + " this user's alone: " + std::strerror(errno);
    ::close(fd);
    if (!why.empty()) {
        err = why;
        return false;
    }
#endif
    return true;
}

bool read_file(const fs::path& p, std::string& out, std::string& err) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        err = "cannot read " + path_text(p);
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

constexpr const char* kSuites = "TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256";

// OpenSSL's queued errors, their reasons in one line (the queue emptied)
std::string ssl_errors() {
    std::string s;
    for (unsigned long e; (e = ERR_get_error()) != 0;) {
        const char* r = ERR_reason_error_string(e);
        char buf[256];
        if (r == nullptr) {
            ERR_error_string_n(e, buf, sizeof buf);
            r = buf;
        }
        if (s.find(r) != std::string::npos) continue;
        if (!s.empty()) s += "; ";
        s += r;
    }
    return s;
}

// what an OpenSSL reason means here, for the log and the main process's error
std::string explain(const std::string& why) {
    auto has = [&](const char* p) { return why.find(p) != std::string::npos; };
    if (has("alert unknown ca") || has("alert bad certificate") || has("alert certificate required") ||
        has("alert certificate unknown") || has("alert access denied"))
        return "the peer does not accept this PC's certificate (" + why +
               "): copy this PC's cert.pem into the peer's peers/ folder";
    // not what a Strata peer sends for a certificate it refuses (those above): nothing in common to agree on
    if (has("alert handshake failure"))
        return "the peer refused the handshake (" + why + "): no TLS 1.3 cipher suite, key exchange or signature "
               "algorithm in common, or it is not a stage peer of this Strata";
    if (has("wrong version number") || has("http request") || has("packet length too long") ||
        has("unsupported protocol"))
        return "the peer does not speak TLS 1.3 (" + why + "): an older Strata, or not a stage peer";
    if (has("peer did not return a certificate"))
        return "the peer sent no certificate (" + why + ")";
    return why;
}

constexpr const char* kNoFingerprint = "SHA256:?";   // X509_digest failed (out of memory)

std::string hex_fingerprint(X509* x) {
    unsigned char md[EVP_MAX_MD_SIZE];
    unsigned n = 0;
    if (X509_digest(x, EVP_sha256(), md, &n) != 1) return kNoFingerprint;
    static const char* hx = "0123456789abcdef";
    std::string s = "SHA256:";
    for (unsigned i = 0; i < n; ++i) {
        s += hx[md[i] >> 4];
        s += hx[md[i] & 15];
    }
    return s;
}

// a certificate's common name for the logs: a refused peer chose it, so printable ASCII only (no line of its own, no
// terminal control or bidi sequence in UTF-8, no character cut in half) and 64 bytes at most; the fingerprint beside
// it names the certificate
std::string common_name(X509* x) {
    char buf[256] = {};
    if (X509_NAME_get_text_by_NID(X509_get_subject_name(x), NID_commonName, buf, sizeof buf) < 0) return "(no name)";
    std::string s;
    for (const char* p = buf; *p != '\0' && s.size() < 64; ++p)
        s += (unsigned char) *p < 0x20 || (unsigned char) *p >= 0x7f ? '?' : *p;
    return s;
}

// a key with a passphrase is refused, not asked for on the terminal
int no_passphrase(char*, int, int, void*) { return 0; }

// the certificates in a PEM text, in order
std::vector<X509*> read_certs(const std::string& pem) {
    std::vector<X509*> v;
    BIO* b = BIO_new_mem_buf(pem.data(), (int) pem.size());
    if (b == nullptr) return v;
    while (X509* x = PEM_read_bio_X509(b, nullptr, no_passphrase, nullptr)) v.push_back(x);
    BIO_free(b);
    ERR_clear_error();   // (the end of the text reads as "no start line")
    return v;
}

// The peer's certificate must be one of those accepted, itself: the chain's check alone also passes one that an
// accepted certificate signed (PARTIAL_CHAIN trusts each as an anchor, a CA's in peers/ too).  A refused certificate's
// name, fingerprint and reason go into the handshake's error (the session's own string).
int verify_peer(int ok, X509_STORE_CTX* xs) {
    SSL* ssl = (SSL*) X509_STORE_CTX_get_ex_data(xs, SSL_get_ex_data_X509_STORE_CTX_idx());
    X509* leaf = X509_STORE_CTX_get0_cert(xs);
    X509* at = X509_STORE_CTX_get_current_cert(xs);   // depth 0: the peer's own (or the store's copy of it)
    if (ok && ssl != nullptr && at != nullptr && X509_STORE_CTX_get_error_depth(xs) == 0) {
        const auto* pinned = (const std::set<std::string>*) SSL_CTX_get_app_data(SSL_get_SSL_CTX(ssl));
        const std::string fp = hex_fingerprint(at);   // "SHA256:?" (no digest) is never a match, whatever was pinned
        if (pinned == nullptr || fp == kNoFingerprint || pinned->count(fp) == 0) {
            X509_STORE_CTX_set_error(xs, X509_V_ERR_CERT_REJECTED);
            ok = 0;
        }
    }
    if (ok) return 1;
    auto* why = ssl != nullptr ? (std::string*) SSL_get_app_data(ssl) : nullptr;
    if (why != nullptr && why->empty() && leaf != nullptr)
        *why = common_name(leaf) + " (" + hex_fingerprint(leaf) + ") is not this PC's nor in its peers/ folder (" +
               X509_verify_cert_error_string(X509_STORE_CTX_get_error(xs)) + ")";
    return 0;
}

// ---- the session's I/O: sealed records go straight into the caller's buffer; fed bytes wait in `in`
struct Io {
    std::vector<char>* out = nullptr;   // a seal's or a handshake's buffer while it runs
    std::vector<char> held;             // written meanwhile (an alert from open, a KeyUpdate's answer)
    std::vector<char> in;
    size_t in_off = 0;
};

int io_write(BIO* b, const char* p, int n) {
    Io* io = (Io*) BIO_get_data(b);
    std::vector<char>& to = io->out != nullptr ? *io->out : io->held;
    to.insert(to.end(), p, p + n);
    return n;
}

int io_read(BIO* b, char* p, int n) {
    Io* io = (Io*) BIO_get_data(b);
    BIO_clear_retry_flags(b);
    const size_t have = io->in.size() - io->in_off;
    if (have == 0) {
        BIO_set_retry_read(b);   // (SSL_ERROR_WANT_READ: the caller feeds more)
        return -1;
    }
    const size_t k = std::min(have, (size_t) n);
    std::memcpy(p, io->in.data() + io->in_off, k);
    io->in_off += k;
    return (int) k;
}

long io_ctrl(BIO*, int cmd, long, void*) {
    switch (cmd) {
    case BIO_CTRL_FLUSH:
    case BIO_CTRL_PUSH:
    case BIO_CTRL_POP:
        return 1;
    default:
        return 0;
    }
}

int io_create(BIO* b) {
    BIO_set_init(b, 1);
    return 1;
}

BIO_METHOD* io_method() {
    static BIO_METHOD* m = [] {
        BIO_METHOD* x = BIO_meth_new(BIO_get_new_index() | BIO_TYPE_SOURCE_SINK, "strata stage link");
        if (x != nullptr) {
            BIO_meth_set_write(x, io_write);
            BIO_meth_set_read(x, io_read);
            BIO_meth_set_ctrl(x, io_ctrl);
            BIO_meth_set_create(x, io_create);
        }
        return x;
    }();
    return m;
}

void add_ext(X509* x, int nid, const char* value) {
    X509V3_CTX c;
    X509V3_set_ctx_nodb(&c);
    X509V3_set_ctx(&c, x, x, nullptr, nullptr, 0);
    if (X509_EXTENSION* e = X509V3_EXT_conf_nid(nullptr, &c, nid, value)) {
        X509_add_ext(x, e, -1);
        X509_EXTENSION_free(e);
    }
}

std::string pem_of(int (*write)(BIO*, void*), void* obj) {
    BIO* b = BIO_new(BIO_s_mem());
    std::string s;
    if (b != nullptr && write(b, obj) == 1) {
        char* p = nullptr;
        const long n = BIO_get_mem_data(b, &p);
        if (n > 0) s.assign(p, (size_t) n);
    }
    BIO_free(b);
    return s;
}

}  // namespace

bool stage_tls_available() { return true; }

struct TlsSession::Impl {
    std::shared_ptr<TlsCredentials> cred;
    SSL* ssl = nullptr;
    Io io;
    std::mutex mu;
    uint64_t sealed = 0;    // since the last KeyUpdate
    std::string refused;    // verify_peer's reason (SSL_get_app_data)
};

TlsCredentials::~TlsCredentials() { SSL_CTX_free((SSL_CTX*) ctx_); }

std::shared_ptr<TlsCredentials> TlsCredentials::load(const std::string& dir, std::string& err) {
    if (dir.empty()) {
        err = "no credentials folder (STRATA_STAGE_TLS): make one on each PC with `strata --stage-keygen DIR` and "
              "copy each PC's DIR/cert.pem into the others' DIR/peers/";
        return nullptr;
    }
    const fs::path d = u8path(dir);
    std::string key_pem, cert_pem;
    if (!read_file(d / "key.pem", key_pem, err) || !read_file(d / "cert.pem", cert_pem, err)) {
        err += " (STRATA_STAGE_TLS=" + dir + "; `strata --stage-keygen DIR` makes them)";
        return nullptr;
    }
    init_openssl();
    ERR_clear_error();
    std::shared_ptr<TlsCredentials> c(new TlsCredentials());
    SSL_CTX* ctx = SSL_CTX_new(TLS_method());
    c->ctx_ = ctx;
    if (ctx == nullptr) {
        err = "TLS: " + ssl_errors();
        return nullptr;
    }
    // TLS 1.3 alone, no resumption (each connection does the whole handshake: both certificates, a fresh key
    // exchange), and nothing sent after the handshake that no message asked for (session tickets)
    bool ok = SSL_CTX_set_min_proto_version(ctx, TLS1_3_VERSION) == 1 &&
              SSL_CTX_set_max_proto_version(ctx, TLS1_3_VERSION) == 1 && SSL_CTX_set_ciphersuites(ctx, kSuites) == 1 &&
              SSL_CTX_set1_groups_list(ctx, "X25519:P-256") == 1 && SSL_CTX_set_num_tickets(ctx, 0) == 1;
    // security level 2 (112-bit keys and up) whatever the OpenSSL build's default is (releases and distributions
    // differ), the level the node agent sets too: the Ed25519 keys and both groups are above it
    SSL_CTX_set_security_level(ctx, 2);
    SSL_CTX_set_options(ctx, SSL_OP_NO_TICKET);
    SSL_CTX_set_session_cache_mode(ctx, SSL_SESS_CACHE_OFF);
    if (!ok) {
        err = "TLS: cannot configure TLS 1.3: " + ssl_errors();
        return nullptr;
    }
    BIO* kb = BIO_new_mem_buf(key_pem.data(), (int) key_pem.size());
    EVP_PKEY* key = kb != nullptr ? PEM_read_bio_PrivateKey(kb, nullptr, no_passphrase, nullptr) : nullptr;
    BIO_free(kb);
    std::vector<X509*> mine = read_certs(cert_pem);
    if (key == nullptr || mine.empty()) {
        err = std::string("TLS: ") + (key == nullptr ? "no key without a passphrase in " : "no certificate in ") +
              path_text(d / (key == nullptr ? "key.pem" : "cert.pem")) + " " + ssl_errors();
        EVP_PKEY_free(key);
        for (X509* x : mine) X509_free(x);
        return nullptr;
    }
    ok = SSL_CTX_use_certificate(ctx, mine[0]) == 1 && SSL_CTX_use_PrivateKey(ctx, key) == 1 &&
         SSL_CTX_check_private_key(ctx) == 1;
    EVP_PKEY_free(key);
    if (!ok) {
        err = "TLS: key.pem and cert.pem do not belong together: " + ssl_errors();
        for (X509* x : mine) X509_free(x);
        return nullptr;
    }
    c->name_ = common_name(mine[0]);
    c->fp_ = hex_fingerprint(mine[0]);
    // the certificates accepted: this PC's own (a main process and a worker on one PC, or one key on every PC) and
    // every one in peers/.  PARTIAL_CHAIN: a certificate found here is trusted itself, nothing above it is looked
    // for; verify_peer then wants the peer's own among them.  NO_CHECK_TIME: a pinned certificate is the PC whatever
    // the clock says (a PC whose clock went back to 2000 would refuse every newer one)
    X509_STORE* st = SSL_CTX_get_cert_store(ctx);
    std::set<std::string> seen;
    auto accept = [&](X509* x) {
        if (seen.insert(hex_fingerprint(x)).second) (void) X509_STORE_add_cert(st, x);
        X509_free(x);
    };
    for (X509* x : mine) accept(x);
    std::error_code ec;
    const fs::path pd = d / "peers";
    if (fs::is_directory(pd, ec)) {
        std::vector<fs::path> files;
        for (const auto& e : fs::directory_iterator(pd, ec)) {
            const std::string ext = path_text(e.path().extension());
            if (e.is_regular_file(ec) && (ext == ".pem" || ext == ".crt")) files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());
        for (const fs::path& f : files) {
            std::string pem;
            if (!read_file(f, pem, err)) return nullptr;
            std::vector<X509*> v = read_certs(pem);
            if (v.empty()) {
                err = "TLS: no certificate in " + path_text(f);
                return nullptr;
            }
            for (X509* x : v) accept(x);
        }
    }
    ERR_clear_error();
    c->accepted_ = (int) seen.size();
    c->pinned_ = std::move(seen);
    SSL_CTX_set_app_data(ctx, &c->pinned_);
    X509_STORE_set_flags(st, X509_V_FLAG_PARTIAL_CHAIN | X509_V_FLAG_NO_CHECK_TIME);
    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, verify_peer);
    return c;
}

// what the keygen's folder may hold: nothing, or what a keygen leaves once its key is removed - a cert.pem and an
// empty peers/ (both made again: stage_keygen). Anything else (another file, a link, a certificate already in peers/,
// a folder that cannot be read) is refused, so the folder never becomes this user's alone with something in it that
// another user put there.
namespace {
bool keygen_folder_ok(const fs::path& d, std::string& err) {
    std::error_code ec;
    if (!fs::exists(d, ec) && !ec) return true;   // (a new folder)
    fs::directory_iterator it(d, ec);
    for (; !ec && it != fs::directory_iterator(); it.increment(ec)) {
        const fs::path n = it->path().filename();
        std::error_code se;
        const fs::file_status s = it->symlink_status(se);
        const bool real = !se && !is_link(it->path());
        if (real && n == "cert.pem" && fs::is_regular_file(s)) continue;
        if (real && n == "peers" && fs::is_directory(s)) {
            const bool empty = fs::is_empty(it->path(), se);
            if (se) {
                err = "cannot read " + path_text(it->path()) + ": " + se.message();
                return false;
            }
            if (empty) continue;
        }
        err = n == "peers" ? path_text(d / "peers") + " is not an empty folder"
                           : path_text(d) + " holds other files (" + path_text(n) + ")";
        err += ": the key needs a folder of its own, a new or an empty one";
        return false;
    }
    if (ec) {
        err = "cannot read " + path_text(d) + ": " + ec.message();
        return false;
    }
    return true;
}

// a file made new, never one written through (one that exists, a link): 0600 or 0644 here, the folder's DACL on
// Windows; nothing left when it fails
bool write_new(const fs::path& p, const std::string& s, int mode) {
#if defined(_WIN32)
    (void) mode;
    const int fd = _wopen(p.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
    bool ok = fd >= 0 && _write(fd, s.data(), (unsigned) s.size()) == (int) s.size();
    if (fd >= 0) ok = _close(fd) == 0 && ok;
#else
    const int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_EXCL, mode);
    bool ok = fd >= 0 && ::write(fd, s.data(), s.size()) == (ssize_t) s.size();
    if (fd >= 0) ok = ::close(fd) == 0 && ok;
#endif
    std::error_code ec;
    if (!ok && fd >= 0) fs::remove(p, ec);
    return ok;
}
}  // namespace

bool stage_keygen(const std::string& dir, const std::string& name, std::string& fingerprint, std::string& err) {
    fs::path d = u8path(dir);
    while (d.has_relative_path() && (!d.has_filename() || d.filename() == ".") && d.parent_path().has_relative_path())
        d = d.parent_path();   // ("keys/", "keys/.": the folder itself, so that a link is seen as one)
    std::error_code ec;
    // refused before the folder's permissions change (owner_only replaces them, on Windows for all it holds too): a
    // link (whoever owns it may point it elsewhere later), a folder with a key, or with anything but a keyless
    // keygen's cert.pem and empty peers/; and by owner_only, another user's folder, or a link swapped in meanwhile
    if (is_link(d)) {
        err = path_text(d) + " is a link: give the folder itself";
        return false;
    }
    if (fs::exists(d / "key.pem", ec)) {
        err = path_text(d / "key.pem") + " exists: not replaced (the PCs that accept its certificate would refuse "
              "this one); for a new key remove it and empty peers/ (its certificates may be copied back after), or "
              "give a new folder";
        return false;
    }
    if (!keygen_folder_ok(d, err)) return false;
    fs::create_directories(d, ec);
    if (ec) {
        err = "cannot make " + path_text(d) + ": " + ec.message();
        return false;
    }
    // (whoever may write the folder may put a certificate into peers/, or swap the key); looked at again once nobody
    // else may write it: what another user put there in between is refused too. Held until the key and its
    // certificate are written (Windows: nobody may move the folder away meanwhile)
    Hold hold;
    if (!owner_only(d, err, hold)) return false;
    if (!keygen_folder_ok(d, err)) return false;
    // what was there is made again, by this user, now that nobody else may write the folder: a cert.pem or a peers/
    // that another user made while it was open to them stays theirs (its owner may open it to them again), and a
    // certificate they add to either would be accepted
    for (const char* n : {"cert.pem", "peers"}) {
        fs::remove(d / n, ec);   // (peers/ only while empty: a file put in it since the look fails here)
        if (ec) {
            err = "cannot remove " + path_text(d / n) + ": " + ec.message();
            return false;
        }
    }
    if (!fs::create_directory(d / "peers", ec)) {
        err = "cannot make " + path_text(d / "peers") + (ec ? ": " + ec.message() : ": it is there again");
        return false;
    }
    init_openssl();
    ERR_clear_error();
    EVP_PKEY* key = nullptr;
    EVP_PKEY_CTX* kc = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr);
    const bool made = kc != nullptr && EVP_PKEY_keygen_init(kc) == 1 && EVP_PKEY_keygen(kc, &key) == 1;
    EVP_PKEY_CTX_free(kc);
    X509* x = made ? X509_new() : nullptr;
    bool ok = x != nullptr;
    if (ok) {
        unsigned char sn[16];
        ok = RAND_bytes(sn, sizeof sn) == 1;
        sn[0] &= 0x7f;   // (a positive serial)
        BIGNUM* bn = ok ? BN_bin2bn(sn, sizeof sn, nullptr) : nullptr;
        ok = ok && bn != nullptr && BN_to_ASN1_INTEGER(bn, X509_get_serialNumber(x)) != nullptr;
        BN_free(bn);
        X509_NAME* nm = X509_get_subject_name(x);
        ok = ok && X509_set_version(x, 2) == 1 && X509_gmtime_adj(X509_getm_notBefore(x), -86400) != nullptr &&
             X509_time_adj_ex(X509_getm_notAfter(x), 36525, 0, nullptr) != nullptr && X509_set_pubkey(x, key) == 1 &&
             X509_NAME_add_entry_by_txt(nm, "O", MBSTRING_ASC, (const unsigned char*) "Strata remote stage", -1, -1,
                                        0) == 1 &&
             X509_NAME_add_entry_by_txt(nm, "CN", MBSTRING_UTF8, (const unsigned char*) name.c_str(), -1, -1, 0) == 1 &&
             X509_set_issuer_name(x, nm) == 1;
        if (ok) {
            add_ext(x, NID_basic_constraints, "critical,CA:FALSE");
            add_ext(x, NID_key_usage, "critical,digitalSignature");
            add_ext(x, NID_ext_key_usage, "serverAuth,clientAuth");
            add_ext(x, NID_subject_key_identifier, "hash");
            ok = X509_sign(x, key, nullptr) > 0;   // (Ed25519 takes no digest)
        }
    }
    std::string key_pem, cert_pem;
    if (ok) {
        key_pem = pem_of([](BIO* b, void* k) {
            return PEM_write_bio_PrivateKey(b, (EVP_PKEY*) k, nullptr, nullptr, 0, nullptr, nullptr);
        }, key);
        cert_pem = pem_of([](BIO* b, void* c) { return PEM_write_bio_X509(b, (X509*) c); }, x);
        fingerprint = hex_fingerprint(x);
        ok = !key_pem.empty() && !cert_pem.empty();
    }
    EVP_PKEY_free(key);
    X509_free(x);
    if (!ok) {
        err = "TLS: cannot make a key: " + ssl_errors();
        return false;
    }
    // the key made only if no key.pem appeared meanwhile, the certificate as new
    const fs::path kp = d / "key.pem";
    if (!write_new(kp, key_pem, 0600)) {
        err = "cannot write " + path_text(kp);
        return false;
    }
    if (!write_new(d / "cert.pem", cert_pem, 0644)) {
        err = "cannot write " + path_text(d / "cert.pem");
        fs::remove(kp, ec);   // (a key without its certificate would only block the next try)
        return false;
    }
    return true;
}

TlsSession::TlsSession(std::shared_ptr<TlsCredentials> cred, bool client) : d_(std::make_unique<Impl>()) {
    d_->cred = std::move(cred);
    if (d_->cred == nullptr || io_method() == nullptr) return;
    SSL* ssl = SSL_new((SSL_CTX*) d_->cred->ctx());
    BIO* b = ssl != nullptr ? BIO_new(io_method()) : nullptr;
    if (b == nullptr) {
        SSL_free(ssl);
        return;
    }
    BIO_set_data(b, &d_->io);
    SSL_set_bio(ssl, b, b);   // (one BIO both ways: SSL_set_bio takes the second reference itself)
    SSL_set_app_data(ssl, &d_->refused);
    SSL_set_read_ahead(ssl, 1);   // (records are read from memory: as much as is there at once)
    if (client) SSL_set_connect_state(ssl);
    else SSL_set_accept_state(ssl);
    d_->ssl = ssl;
}

TlsSession::~TlsSession() { SSL_free(d_->ssl); }

TlsSession::Step TlsSession::handshake(std::vector<char>& out, std::string& err) {
    std::lock_guard<std::mutex> lk(d_->mu);
    if (d_->ssl == nullptr) {
        err = "TLS: no session (no credentials)";
        return Step::Failed;
    }
    ERR_clear_error();
    out.insert(out.end(), d_->io.held.begin(), d_->io.held.end());
    d_->io.held.clear();
    d_->io.out = &out;
    const int r = SSL_do_handshake(d_->ssl);
    d_->io.out = nullptr;
    if (r == 1) {
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
        X509* pc = SSL_get1_peer_certificate(d_->ssl);
#else
        X509* pc = SSL_get_peer_certificate(d_->ssl);
#endif
        peer_ = pc != nullptr ? common_name(pc) + " (" + stage_tls_short(hex_fingerprint(pc)) + ")"
                              : std::string("(no certificate)");
        peer_ += std::string(", ") + SSL_get_version(d_->ssl) + " " + SSL_get_cipher_name(d_->ssl);
        X509_free(pc);
        return Step::Done;
    }
    const int e = SSL_get_error(d_->ssl, r);
    if (e == SSL_ERROR_WANT_READ || e == SSL_ERROR_WANT_WRITE) return Step::More;
    const std::string why = ssl_errors();
    err = "TLS: " + (!d_->refused.empty() ? d_->refused : explain(why.empty() ? "the handshake failed" : why));
    return Step::Failed;
}

void TlsSession::feed(const void* p, size_t n) {
    std::lock_guard<std::mutex> lk(d_->mu);
    Io& io = d_->io;
    if (io.in_off == io.in.size()) {
        io.in.clear();
        io.in_off = 0;
    }
    io.in.insert(io.in.end(), (const char*) p, (const char*) p + n);
}

bool TlsSession::seal(const void* p, size_t n, std::vector<char>& out, std::string& err) {
    std::lock_guard<std::mutex> lk(d_->mu);
    if (d_->ssl == nullptr) {
        err = "TLS: no session";
        return false;
    }
    ERR_clear_error();
    out.insert(out.end(), d_->io.held.begin(), d_->io.held.end());
    d_->io.held.clear();
    const uint64_t every = g_rekey_bytes.load() != 0 ? g_rekey_bytes.load() : kRekeyBytes;
    if (d_->sealed >= every) {
        // this side's next key (the peer updates its reading key; it is not asked to update its own)
        if (SSL_key_update(d_->ssl, SSL_KEY_UPDATE_NOT_REQUESTED) != 1) {
            err = "TLS: KeyUpdate: " + ssl_errors();
            return false;
        }
        d_->sealed = 0;
        ++g_key_updates;
    }
    d_->io.out = &out;
    size_t w = 0;
    const int r = n > 0 ? SSL_write_ex(d_->ssl, p, n, &w) : 1;
    d_->io.out = nullptr;
    if (r != 1 || w != n) {
        err = "TLS: " + explain(ssl_errors());
        return false;
    }
    d_->sealed += n;
    return true;
}

long long TlsSession::open(void* p, size_t n, std::string& err) {
    std::lock_guard<std::mutex> lk(d_->mu);
    if (d_->ssl == nullptr) {
        err = "TLS: no session";
        return -1;
    }
    ERR_clear_error();
    size_t got = 0;
    if (SSL_read_ex(d_->ssl, p, n, &got) == 1) return (long long) got;
    const int e = SSL_get_error(d_->ssl, 0);
    if (e == SSL_ERROR_WANT_READ) return 0;
    if (e == SSL_ERROR_ZERO_RETURN) {
        err = "TLS: the peer ended the session";
        return -2;
    }
    const std::string why = ssl_errors();
    err = "TLS: " + explain(why.empty() ? "the records failed" : why);
    return -1;
}

bool TlsSession::buffered() {
    std::lock_guard<std::mutex> lk(d_->mu);
    return d_->io.in_off < d_->io.in.size() || (d_->ssl != nullptr && SSL_has_pending(d_->ssl) == 1);
}

#else   // ---- a build without OpenSSL: no remote stage

namespace {
constexpr const char* kNoTls = "this build has no TLS (it was built without OpenSSL 1.1.1 or later); the remote "
                               "stage needs it: build again with OpenSSL's development files (Ubuntu / Debian: "
                               "libssl-dev; the CMake message names what was missing)";
}

bool stage_tls_available() { return false; }

struct TlsSession::Impl {};

TlsCredentials::~TlsCredentials() = default;

std::shared_ptr<TlsCredentials> TlsCredentials::load(const std::string&, std::string& err) {
    err = kNoTls;
    return nullptr;
}

bool stage_keygen(const std::string&, const std::string&, std::string&, std::string& err) {
    err = kNoTls;
    return false;
}

TlsSession::TlsSession(std::shared_ptr<TlsCredentials>, bool) : d_(std::make_unique<Impl>()) {}
TlsSession::~TlsSession() = default;
TlsSession::Step TlsSession::handshake(std::vector<char>&, std::string& err) {
    err = kNoTls;
    return Step::Failed;
}
void TlsSession::feed(const void*, size_t) {}
bool TlsSession::seal(const void*, size_t, std::vector<char>&, std::string& err) {
    err = kNoTls;
    return false;
}
long long TlsSession::open(void*, size_t, std::string& err) {
    err = kNoTls;
    return -1;
}
bool TlsSession::buffered() { return false; }

#endif

}  // namespace strata::net
