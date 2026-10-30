// include/strata/net/stage_tls.hpp - the remote stage's TLS 1.3: this PC's key, the PCs it accepts, and one
// connection's records, in memory.
//
// Every connection between the PCs of a split is TLS 1.3, and both ends prove who they are.  Each PC has a key of its
// own (`strata --stage-keygen DIR` writes DIR/key.pem, which never leaves the PC) and its certificate (DIR/cert.pem).
// A PC accepts a peer only when the certificate it presents is its own or one of DIR/peers/*.pem (the other PCs'
// cert.pem, copied there), that certificate itself: one these certificates signed is not accepted (pinned, no
// certificate authority).  The main process so knows that its worker is one of its PCs, and a worker that its main
// process is.  Host names, addresses and the clock are not looked at - the certificate is the PC - and there is no
// plain-text mode.  The key exchange is X25519 (each connection's keys are its own: a key copied later does not open a recorded
// session), the records AES-128-GCM (AES-256-GCM, ChaCha20-Poly1305 if a side lacks it).
//
// A TlsSession holds no socket.  Its owner sends what it produces and feeds it what arrives, so the sockets keep
// their own deadlines, stop flags and shutdowns (stage_link.cpp).  One thread may seal while another opens (a lock
// inside, held for one piece's encryption or one record's decryption); one thread at a time for each.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace strata::net {

/// Whether this build has TLS (OpenSSL 1.1.1 or later).  Without it a remote stage does not start.
bool stage_tls_available();

/// The credentials folder, STRATA_STAGE_TLS (empty: not set).
std::string stage_tls_dir();

/// A new key and its certificate in `dir` (made if missing, then this user's alone: mode 0700, on Windows a DACL of
/// this user, SYSTEM and Administrators only): key.pem (Ed25519), cert.pem (self-signed, `name` as its common name)
/// and an empty peers/, each made new.  A link (on Windows a junction too), another user's folder (on Windows one
/// whose owner is neither this user nor the Administrators), on Windows a folder another process holds with the right
/// to move it (the keygen holds it so until it has written), a folder that has a key.pem, and one with anything but
/// what a keygen left once its key was removed (a cert.pem and an empty peers/) are refused before their permissions
/// change; what another user puts in the folder before it is this user's alone is refused after.  `fingerprint`: the
/// certificate's SHA-256 ("SHA256:" and 64 hex digits).
bool stage_keygen(const std::string& dir, const std::string& name, std::string& fingerprint, std::string& err);

/// This PC's name for --stage-keygen's default: the host name ("strata" when there is none).
std::string stage_host_name();

/// One process's credentials: its key and certificate, and the certificates it accepts.  Its connections share it.
class TlsCredentials {
public:
    ~TlsCredentials();
    TlsCredentials(const TlsCredentials&) = delete;
    TlsCredentials& operator=(const TlsCredentials&) = delete;
    /// DIR/key.pem, DIR/cert.pem and every certificate in DIR/peers/*.pem (a file may hold several).
    static std::shared_ptr<TlsCredentials> load(const std::string& dir, std::string& err);
    const std::string& name() const { return name_; }          ///< this PC's certificate's common name
    const std::string& fingerprint() const { return fp_; }     ///< its SHA-256
    int accepted() const { return accepted_; }                 ///< certificates accepted, this PC's own included
    void* ctx() const { return ctx_; }                         ///< (the SSL_CTX)

private:
    TlsCredentials() = default;
    void* ctx_ = nullptr;
    std::string name_, fp_;
    int accepted_ = 0;
    std::set<std::string> pinned_;   // the accepted certificates' fingerprints (the handshake checks the peer's)
};

/// One connection's TLS state.
class TlsSession {
public:
    TlsSession(std::shared_ptr<TlsCredentials> cred, bool client);
    ~TlsSession();
    TlsSession(const TlsSession&) = delete;
    TlsSession& operator=(const TlsSession&) = delete;

    enum class Step { Done, More, Failed };
    /// The handshake's next step: what to send is appended to `out` (also on a failure: the alert that tells the
    /// peer why).  More: send it, feed() what arrives, call again.  Failed: `err` says why (a certificate this PC
    /// does not accept names its fingerprint).
    Step handshake(std::vector<char>& out, std::string& err);
    /// Bytes from the peer.
    void feed(const void* p, size_t n);
    /// `n` bytes as records, appended to `out`.  Every 16 GiB this side sends a KeyUpdate first (TLS 1.3 limits how
    /// many records one AES-GCM key may seal).
    bool seal(const void* p, size_t n, std::vector<char>& out, std::string& err);
    /// Up to `n` bytes of what the peer sent (one record's at most): their count; 0: more must be fed first; -1: the
    /// records failed (`err`: why - a forged or damaged record, an alert); -2: the peer ended the session.
    long long open(void* p, size_t n, std::string& err);
    /// Whether bytes fed are not opened yet (a whole or part record, or a record's rest): a reader must look here
    /// before it waits on the socket.
    bool buffered();
    /// After the handshake: the peer's certificate (its name and fingerprint), the version and the cipher.
    const std::string& peer() const { return peer_; }

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
    std::string peer_;
};

/// A fingerprint's short form in logs: "SHA256:" and the first 16 hex digits.
std::string stage_tls_short(const std::string& fingerprint);

/// Tests: the bytes each side seals between KeyUpdates (0: the default, 16 GiB), and the KeyUpdates sent so far by
/// this process.
void stage_tls_set_rekey_bytes(uint64_t n);
uint64_t stage_tls_key_updates();

}  // namespace strata::net
