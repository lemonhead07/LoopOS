#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

namespace LoopOS {
namespace Utils {

template <typename T, std::size_t Alignment>
class AlignedAllocator {
public:
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }

        const std::size_t bytes = n * sizeof(T);
        void* ptr = nullptr;

#if defined(_MSC_VER)
        ptr = _aligned_malloc(bytes, Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix__)
        const std::size_t aligned_bytes = ((bytes + Alignment - 1) / Alignment) * Alignment;
        if (posix_memalign(&ptr, Alignment, aligned_bytes) != 0) {
            throw std::bad_alloc();
        }
#else
        const std::size_t aligned_bytes = ((bytes + Alignment - 1) / Alignment) * Alignment;
        ptr = std::aligned_alloc(Alignment, aligned_bytes);
        if (!ptr) {
            throw std::bad_alloc();
        }
#endif

        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        if (!ptr) {
            return;
        }

#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
};

template <typename T, std::size_t Alignment>
inline bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<T, Alignment>&) noexcept {
    return true;
}

template <typename T, typename U, std::size_t Alignment>
inline bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) noexcept {
    return true;
}

template <typename T, std::size_t Alignment>
inline bool operator!=(const AlignedAllocator<T, Alignment>& a, const AlignedAllocator<T, Alignment>& b) noexcept {
    return !(a == b);
}

template <typename T, typename U, std::size_t Alignment>
inline bool operator!=(const AlignedAllocator<T, Alignment>& a, const AlignedAllocator<U, Alignment>& b) noexcept {
    return !(a == b);
}

} // namespace Utils
} // namespace LoopOS
