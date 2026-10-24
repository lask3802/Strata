// A remote stage's worker keeps its layers' part of each of the main process's slots (ds4_stage.hpp) here, under the
// main process's slot id: in RAM up to `ram_max` parts, the least recently used past that written to a folder, the
// folder's bytes capped, as the main process bounds its own slots (STRATA_DS4_SLOT_CAP / _GB / _DIR in ds4_main.cpp).
// The main process alone decides what lives: every CkptSave / CkptRestore / Reset names the ids it holds (`keep`), and
// prune() drops the rest; the store evicts on its own only to stay inside its two bounds, and a part it dropped is
// simply missing when the main process asks for it (the main process then reads the conversation from token 0).
// A worker's parts belong to the connection that made them (the main process's ids are its own): clear() at a
// disconnect, and a folder left by an earlier run is emptied when the store is made.  No GPU: tests/dsv41/stage_test.cpp.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace strata::ds4::stage {

class PartStore {
public:
    /// `dir`: where parts past `ram_max` go (made when first needed; "": none - a part past the RAM is dropped);
    /// `disk_budget`: the folder's bytes at most (0: no folder), never counting the part just put
    PartStore(std::string dir, size_t ram_max, uint64_t disk_budget)
        : dir_(std::move(dir)), ram_max_(ram_max), budget_(dir_.empty() ? 0 : disk_budget) {
        if (dir_.empty()) return;
        nonce_ = (uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();   // (two stores on one folder never share a file)
        std::error_code ec;   // (an earlier run's parts: another id space)
        for (const auto& de : std::filesystem::directory_iterator(dir_, ec)) {
            const std::string fn = de.path().filename().string();
            if (fn.rfind("part-", 0) == 0) std::filesystem::remove(de.path(), ec);
        }
    }
    ~PartStore() { clear(); }
    PartStore(const PartStore&) = delete;
    PartStore& operator=(const PartStore&) = delete;

    /// `bytes` as the part of `id` at position `len` (in place of one of the same id).  False: not held (`why`) - there
    /// is no room for it in RAM nor in the folder (an earlier part of that id is gone too).  Other parts may leave to
    /// make room.
    bool put(int64_t id, int64_t len, std::vector<uint8_t>&& bytes, std::string& why) {
        if (bytes.empty()) {
            why = "an empty part";
            return false;
        }
        if (ram_max_ == 0 && budget_ == 0) {
            why = "this worker keeps no slots (STRATA_DS4_SLOT_CAP and STRATA_DS4_SLOT_GB are 0)";
            return false;
        }
        drop_(id);
        Part p;
        p.len = len;
        p.size = bytes.size();
        p.used = ++clock_;
        p.ram = std::move(bytes);
        parts_[id] = std::move(p);
        while (ram_count_() > ram_max_) {   // past the RAM cap: the least recently used part (never the new one) leaves RAM
            int64_t lru = 0;
            bool have = false;
            for (const auto& kv : parts_)
                if (kv.first != id && !kv.second.ram.empty() && (!have || kv.second.used < parts_[lru].used)) {
                    lru = kv.first;
                    have = true;
                }
            if (!have) break;
            if (!spill_(lru)) drop_(lru);
        }
        if (ram_count_() > ram_max_) {   // (ram_max 0: the new part itself goes to the folder)
            if (!spill_(id)) {
                why = "no room in the folder";
                drop_(id);
                return false;
            }
        }
        trim_(id);
        return true;
    }

    enum class Get { Ok, Missing, Failed };
    /// the part of `id` back into `out`, if it was put at position `len`.  Missing: not held (never put, pruned or
    /// dropped); Failed: held but not at `len`, or its file cannot be read (it is dropped then)
    Get get(int64_t id, int64_t len, std::vector<uint8_t>& out, std::string& why) {
        const auto it = parts_.find(id);
        if (it == parts_.end()) return Get::Missing;
        Part& p = it->second;
        if (p.len != len) {
            why = "part " + std::to_string(id) + " was saved at " + std::to_string(p.len) + ", asked for at " +
                  std::to_string(len);
            return Get::Failed;
        }
        p.used = ++clock_;
        if (!p.ram.empty()) {
            out = p.ram;
            return Get::Ok;
        }
        std::ifstream f(p.path, std::ios::binary);
        out.resize(p.size);
        if (!f.read((char*) out.data(), (std::streamsize) p.size)) {
            why = "reading " + p.path + " failed";
            drop_(id);
            out.clear();
            return Get::Failed;
        }
        return Get::Ok;
    }

    /// drops every part not in `keep`
    void prune(const std::vector<int64_t>& keep) {
        std::vector<int64_t> gone;
        for (const auto& kv : parts_)
            if (std::find(keep.begin(), keep.end(), kv.first) == keep.end()) gone.push_back(kv.first);
        for (int64_t id : gone) drop_(id);
    }
    void clear() {
        std::vector<int64_t> all;
        for (const auto& kv : parts_) all.push_back(kv.first);
        for (int64_t id : all) drop_(id);
    }

    /// the file holding the part of `id` ("": it is in RAM, or not held)
    std::string file_of(int64_t id) const {
        const auto it = parts_.find(id);
        return it == parts_.end() ? std::string() : it->second.path;
    }
    bool has(int64_t id) const { return parts_.count(id) != 0; }
    size_t size() const { return parts_.size(); }
    size_t ram_parts() const { return ram_count_(); }
    size_t disk_parts() const { return parts_.size() - ram_count_(); }
    uint64_t disk_bytes() const {
        uint64_t n = 0;
        for (const auto& kv : parts_)
            if (kv.second.ram.empty()) n += kv.second.size;
        return n;
    }

private:
    struct Part {
        int64_t len = 0;
        uint64_t size = 0, used = 0;
        std::vector<uint8_t> ram;   // empty: in the folder (path)
        std::string path;
    };
    size_t ram_count_() const {
        size_t n = 0;
        for (const auto& kv : parts_) n += kv.second.ram.empty() ? 0 : 1;
        return n;
    }
    void drop_(int64_t id) {
        const auto it = parts_.find(id);
        if (it == parts_.end()) return;
        if (it->second.ram.empty() && !it->second.path.empty()) {
            std::error_code ec;
            std::filesystem::remove(it->second.path, ec);
        }
        parts_.erase(it);
    }
    // a part in RAM into the folder (written under a temporary name and renamed); false: not written
    bool spill_(int64_t id) {
        Part& p = parts_[id];
        if (budget_ == 0 || p.ram.empty()) return false;
        std::error_code ec;
        std::filesystem::create_directories(dir_, ec);
        const std::string path = (std::filesystem::path(dir_) / ("part-" + std::to_string(nonce_) + "-" + std::to_string(id) + ".bin")).string();
        {
            std::ofstream f(path + ".tmp", std::ios::binary | std::ios::trunc);
            f.write((const char*) p.ram.data(), (std::streamsize) p.ram.size());
            f.flush();
            if (!f) {
                std::filesystem::remove(path + ".tmp", ec);
                return false;
            }
        }
        std::filesystem::rename(path + ".tmp", path, ec);
        if (ec) {
            std::filesystem::remove(path + ".tmp", ec);
            return false;
        }
        p.path = path;
        p.ram.clear();
        p.ram.shrink_to_fit();
        return true;
    }
    // the folder over its budget: its least recently used part goes, never `keep` (a part bigger than the budget stays)
    void trim_(int64_t keep) {
        while (disk_bytes() > budget_) {
            int64_t lru = 0;
            bool have = false;
            for (const auto& kv : parts_)
                if (kv.first != keep && kv.second.ram.empty() && (!have || kv.second.used < parts_[lru].used)) {
                    lru = kv.first;
                    have = true;
                }
            if (!have) break;
            drop_(lru);
        }
    }

    std::string dir_;
    size_t ram_max_;
    uint64_t budget_;
    uint64_t nonce_ = 0;   // (in the names of this store's files)
    uint64_t clock_ = 0;
    std::map<int64_t, Part> parts_;
};

}  // namespace strata::ds4::stage
