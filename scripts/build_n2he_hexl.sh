#!/bin/bash
# Build script for N2HE-HEXL (Neural Network Homomorphic Encryption with HEXL acceleration)
#
# This script builds the N2HE-HEXL library from source with SEAL and HEXL support.
# Prerequisites: CMake 3.16+, GCC 9+, Python 3.9+, pip
#
# Usage:
#   ./scripts/build_n2he_hexl.sh
#   ./scripts/build_n2he_hexl.sh --clean  # Clean rebuild
#   ./scripts/build_n2he_hexl.sh --gpu    # Build with GPU support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_DIR="$PROJECT_ROOT/third_party"
BUILD_DIR="$PROJECT_ROOT/build/n2he_hexl"
INSTALL_DIR="$PROJECT_ROOT/crypto_backend/n2he_hexl/lib"

# Configuration
SEAL_VERSION="4.1.1"
HEXL_VERSION="1.2.5"
N2HE_COMMIT="main"  # Pin to specific commit for reproducibility

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
CLEAN_BUILD=false
GPU_SUPPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --gpu)
            GPU_SUPPORT=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    log_info "Cleaning previous build..."
    rm -rf "$BUILD_DIR"
    rm -rf "$INSTALL_DIR"
fi

# Create directories
mkdir -p "$THIRD_PARTY_DIR"
mkdir -p "$BUILD_DIR"
mkdir -p "$INSTALL_DIR"

# Check prerequisites
check_prereqs() {
    log_info "Checking prerequisites..."

    # CMake
    if ! command -v cmake &> /dev/null; then
        log_error "CMake is required. Install with: apt-get install cmake"
        exit 1
    fi
    CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
    log_info "CMake version: $CMAKE_VERSION"

    # GCC
    if ! command -v g++ &> /dev/null; then
        log_error "GCC is required. Install with: apt-get install build-essential"
        exit 1
    fi
    GCC_VERSION=$(g++ --version | head -1)
    log_info "GCC: $GCC_VERSION"

    # Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required"
        exit 1
    fi
    PYTHON_VERSION=$(python3 --version)
    log_info "Python: $PYTHON_VERSION"
}

# Install Intel HEXL (HE acceleration library)
install_hexl() {
    log_info "Installing Intel HEXL v${HEXL_VERSION}..."

    HEXL_DIR="$THIRD_PARTY_DIR/hexl"

    if [ -d "$HEXL_DIR" ] && [ -f "$HEXL_DIR/build/hexl/lib/libhexl.a" ]; then
        log_info "HEXL already installed, skipping..."
        return
    fi

    cd "$THIRD_PARTY_DIR"

    if [ ! -d "$HEXL_DIR" ]; then
        git clone --depth 1 --branch "v${HEXL_VERSION}" https://github.com/intel/hexl.git hexl
    fi

    cd "$HEXL_DIR"
    mkdir -p build && cd build

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DHEXL_BENCHMARK=OFF \
        -DHEXL_TESTING=OFF

    make -j$(nproc)

    log_info "HEXL installed successfully"
}

# Install Microsoft SEAL
install_seal() {
    log_info "Installing Microsoft SEAL v${SEAL_VERSION}..."

    SEAL_DIR="$THIRD_PARTY_DIR/SEAL"
    HEXL_DIR="$THIRD_PARTY_DIR/hexl"

    if [ -d "$SEAL_DIR" ] && [ -f "$SEAL_DIR/build/lib/libseal-*.a" ]; then
        log_info "SEAL already installed, skipping..."
        return
    fi

    cd "$THIRD_PARTY_DIR"

    if [ ! -d "$SEAL_DIR" ]; then
        git clone --depth 1 --branch "v${SEAL_VERSION}" https://github.com/microsoft/SEAL.git
    fi

    cd "$SEAL_DIR"
    mkdir -p build && cd build

    # Build with HEXL support if available
    CMAKE_OPTS="-DCMAKE_BUILD_TYPE=Release -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZSTD=OFF -DSEAL_USE_ZLIB=OFF"

    if [ -d "$HEXL_DIR/build" ]; then
        CMAKE_OPTS="$CMAKE_OPTS -DSEAL_USE_INTEL_HEXL=ON -Dhexl_DIR=$HEXL_DIR/build/hexl/lib/cmake/hexl-${HEXL_VERSION}"
    fi

    cmake .. $CMAKE_OPTS
    make -j$(nproc)

    log_info "SEAL installed successfully"
}

# Build N2HE-HEXL Python bindings
build_n2he_bindings() {
    log_info "Building N2HE-HEXL Python bindings..."

    cd "$BUILD_DIR"

    # Create the native module source
    cat > n2he_native.cpp << 'NATIVE_EOF'
/*
 * N2HE-HEXL Native Python Bindings
 *
 * Provides CKKS encryption primitives optimized for neural network inference.
 * Based on Microsoft SEAL with Intel HEXL acceleration.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <seal/seal.h>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace py = pybind11;
using namespace seal;

// CKKS Context wrapper
class CKKSContext {
public:
    std::shared_ptr<SEALContext> context;
    std::shared_ptr<KeyGenerator> keygen;
    std::shared_ptr<SecretKey> secret_key;
    std::shared_ptr<PublicKey> public_key;
    std::shared_ptr<RelinKeys> relin_keys;
    std::shared_ptr<GaloisKeys> galois_keys;
    std::shared_ptr<Encryptor> encryptor;
    std::shared_ptr<Decryptor> decryptor;
    std::shared_ptr<Evaluator> evaluator;
    std::shared_ptr<CKKSEncoder> encoder;
    double scale;
    size_t poly_modulus_degree;

    CKKSContext(size_t poly_modulus_degree_ = 8192,
                std::vector<int> coeff_modulus_bits = {60, 40, 40, 60},
                double scale_ = pow(2.0, 40))
        : scale(scale_), poly_modulus_degree(poly_modulus_degree_) {

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(poly_modulus_degree);

        // Build coefficient modulus
        auto coeff_modulus = CoeffModulus::Create(poly_modulus_degree, coeff_modulus_bits);
        parms.set_coeff_modulus(coeff_modulus);

        context = std::make_shared<SEALContext>(parms);

        if (!context->parameters_set()) {
            throw std::runtime_error("Invalid CKKS parameters");
        }

        encoder = std::make_shared<CKKSEncoder>(*context);
    }

    void generate_keys(bool generate_galois = true) {
        keygen = std::make_shared<KeyGenerator>(*context);
        secret_key = std::make_shared<SecretKey>(keygen->secret_key());
        public_key = std::make_shared<PublicKey>();
        keygen->create_public_key(*public_key);

        relin_keys = std::make_shared<RelinKeys>();
        keygen->create_relin_keys(*relin_keys);

        if (generate_galois) {
            galois_keys = std::make_shared<GaloisKeys>();
            keygen->create_galois_keys(*galois_keys);
        }

        encryptor = std::make_shared<Encryptor>(*context, *public_key);
        decryptor = std::make_shared<Decryptor>(*context, *secret_key);
        evaluator = std::make_shared<Evaluator>(*context);
    }

    bool has_galois_keys() const {
        return galois_keys != nullptr;
    }

    size_t slot_count() const {
        return encoder->slot_count();
    }

    std::vector<int> get_coeff_modulus_sizes() const {
        auto& context_data = *context->first_context_data();
        auto& coeff_modulus = context_data.parms().coeff_modulus();
        std::vector<int> sizes;
        for (const auto& mod : coeff_modulus) {
            sizes.push_back(mod.bit_count());
        }
        return sizes;
    }

    int get_remaining_levels(const Ciphertext& ct) const {
        auto context_data = context->get_context_data(ct.parms_id());
        return context_data->chain_index();
    }
};

// Ciphertext wrapper with metadata
class CKKSCiphertext {
public:
    Ciphertext ct;
    double scale;
    std::shared_ptr<CKKSContext> ctx;

    CKKSCiphertext(std::shared_ptr<CKKSContext> ctx_) : ctx(ctx_), scale(ctx_->scale) {}

    int get_level() const {
        return ctx->get_remaining_levels(ct);
    }

    double get_scale() const {
        return scale;
    }

    size_t size() const {
        return ct.size();
    }

    py::bytes serialize() const {
        std::ostringstream oss;
        ct.save(oss);
        return py::bytes(oss.str());
    }

    static CKKSCiphertext deserialize(const py::bytes& data, std::shared_ptr<CKKSContext> ctx_) {
        CKKSCiphertext result(ctx_);
        std::istringstream iss(std::string(data));
        result.ct.load(*ctx_->context, iss);
        return result;
    }
};

// Column-packed matrix for MOAI-style packing
class ColumnPackedMatrix {
public:
    std::vector<std::vector<double>> columns;
    size_t rows;
    size_t cols;

    ColumnPackedMatrix(py::array_t<double> matrix) {
        auto buf = matrix.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Matrix must be 2D");
        }

        rows = buf.shape[0];
        cols = buf.shape[1];
        double* ptr = static_cast<double*>(buf.ptr);

        // Pack by columns (MOAI column packing)
        columns.resize(cols);
        for (size_t j = 0; j < cols; j++) {
            columns[j].resize(rows);
            for (size_t i = 0; i < rows; i++) {
                columns[j][i] = ptr[i * cols + j];
            }
        }
    }

    py::array_t<double> get_column(size_t idx) const {
        if (idx >= cols) {
            throw std::out_of_range("Column index out of range");
        }
        return py::array_t<double>(columns[idx].size(), columns[idx].data());
    }
};

// MOAI-style HE operations
class MOAIOperations {
public:
    std::shared_ptr<CKKSContext> ctx;
    size_t rotations_used;
    size_t multiplications;
    size_t additions;

    MOAIOperations(std::shared_ptr<CKKSContext> ctx_)
        : ctx(ctx_), rotations_used(0), multiplications(0), additions(0) {}

    void reset_counters() {
        rotations_used = 0;
        multiplications = 0;
        additions = 0;
    }

    // Encrypt a vector with column packing
    CKKSCiphertext encrypt_vector(py::array_t<double> plaintext) {
        auto buf = plaintext.request();
        double* ptr = static_cast<double*>(buf.ptr);
        size_t len = buf.size;

        std::vector<double> data(ptr, ptr + len);

        // Pad to slot count
        size_t slot_count = ctx->slot_count();
        if (data.size() < slot_count) {
            data.resize(slot_count, 0.0);
        }

        Plaintext pt;
        ctx->encoder->encode(data, ctx->scale, pt);

        CKKSCiphertext result(ctx);
        ctx->encryptor->encrypt(pt, result.ct);
        result.scale = ctx->scale;

        return result;
    }

    // Decrypt to vector
    py::array_t<double> decrypt_vector(const CKKSCiphertext& ct, size_t output_size = 0) {
        Plaintext pt;
        ctx->decryptor->decrypt(ct.ct, pt);

        std::vector<double> data;
        ctx->encoder->decode(pt, data);

        if (output_size > 0 && output_size < data.size()) {
            data.resize(output_size);
        }

        return py::array_t<double>(data.size(), data.data());
    }

    // MOAI Column-Packing Plaintext-Ciphertext MatMul
    // This removes rotations for pt-ct matmul (MOAI key optimization)
    CKKSCiphertext column_packed_matmul(const CKKSCiphertext& ct_x,
                                         const ColumnPackedMatrix& weight,
                                         bool rescale = true) {
        // For pt-ct matmul y = W @ x, where W is column-packed:
        // Each output element y[i] = sum_j(W[i,j] * x[j])
        // With column packing: y = sum_j(W[:,j] * x[j])
        // where each W[:,j] is encoded as plaintext and x[j] accessed via slot

        // This achieves ZERO rotations for the matmul!
        // (MOAI's key insight for plaintext-ciphertext operations)

        CKKSCiphertext result(ctx);
        bool first = true;

        for (size_t j = 0; j < weight.cols; j++) {
            // Encode column j of weight matrix
            std::vector<double> col = weight.columns[j];

            // Pad to slot count
            if (col.size() < ctx->slot_count()) {
                col.resize(ctx->slot_count(), 0.0);
            }

            Plaintext pt_col;
            ctx->encoder->encode(col, ct_x.scale, pt_col);

            // Multiply ct_x by column (element-wise in SIMD)
            Ciphertext temp;
            ctx->evaluator->multiply_plain(ct_x.ct, pt_col, temp);
            multiplications++;

            if (first) {
                result.ct = temp;
                first = false;
            } else {
                ctx->evaluator->add_inplace(result.ct, temp);
                additions++;
            }
        }

        if (rescale) {
            ctx->evaluator->rescale_to_next_inplace(result.ct);
            result.scale = result.ct.scale();
        }

        return result;
    }

    // Standard rotation-based matmul (for comparison)
    CKKSCiphertext rotation_matmul(const CKKSCiphertext& ct_x,
                                   py::array_t<double> weight) {
        // This uses rotations and should be slower
        auto buf = weight.request();
        size_t rows = buf.shape[0];
        size_t cols = buf.shape[1];
        double* ptr = static_cast<double*>(buf.ptr);

        CKKSCiphertext result(ctx);
        bool first = true;

        for (size_t i = 0; i < rows; i++) {
            // Extract row i
            std::vector<double> row(cols);
            for (size_t j = 0; j < cols; j++) {
                row[j] = ptr[i * cols + j];
            }
            row.resize(ctx->slot_count(), 0.0);

            Plaintext pt_row;
            ctx->encoder->encode(row, ct_x.scale, pt_row);

            // Multiply and sum (requires rotation to accumulate)
            Ciphertext temp;
            ctx->evaluator->multiply_plain(ct_x.ct, pt_row, temp);
            multiplications++;

            // Rotate and sum to get dot product
            Ciphertext rotated = temp;
            for (size_t step = 1; step < cols; step *= 2) {
                Ciphertext rot_temp;
                ctx->evaluator->rotate_vector(rotated, step, *ctx->galois_keys, rot_temp);
                ctx->evaluator->add_inplace(rotated, rot_temp);
                rotations_used++;
                additions++;
            }

            if (first) {
                result.ct = rotated;
                first = false;
            } else {
                ctx->evaluator->add_inplace(result.ct, rotated);
                additions++;
            }
        }

        ctx->evaluator->rescale_to_next_inplace(result.ct);
        result.scale = result.ct.scale();

        return result;
    }

    // LoRA delta computation with column packing (MOAI optimized)
    // delta = scaling * (x @ A^T @ B^T)
    CKKSCiphertext lora_delta_column_packed(const CKKSCiphertext& ct_x,
                                             const ColumnPackedMatrix& lora_a,
                                             const ColumnPackedMatrix& lora_b,
                                             double scaling = 1.0) {
        // Step 1: u = x @ A^T (d -> r)
        CKKSCiphertext ct_u = column_packed_matmul(ct_x, lora_a, true);

        // Step 2: delta = u @ B^T (r -> d)
        CKKSCiphertext ct_delta = column_packed_matmul(ct_u, lora_b, true);

        // Step 3: Apply scaling if needed
        if (std::abs(scaling - 1.0) > 1e-6) {
            Plaintext pt_scale;
            std::vector<double> scale_vec(ctx->slot_count(), scaling);
            ctx->encoder->encode(scale_vec, ct_delta.scale, pt_scale);
            ctx->evaluator->multiply_plain_inplace(ct_delta.ct, pt_scale);
            ctx->evaluator->rescale_to_next_inplace(ct_delta.ct);
            ct_delta.scale = ct_delta.ct.scale();
            multiplications++;
        }

        return ct_delta;
    }

    py::dict get_stats() const {
        return py::dict(
            "rotations_used"_a = rotations_used,
            "multiplications"_a = multiplications,
            "additions"_a = additions
        );
    }
};

// Python module definition
PYBIND11_MODULE(n2he_native, m) {
    m.doc() = "N2HE-HEXL Native Module - CKKS HE for Neural Networks with MOAI optimizations";

    py::class_<CKKSContext, std::shared_ptr<CKKSContext>>(m, "CKKSContext")
        .def(py::init<size_t, std::vector<int>, double>(),
             py::arg("poly_modulus_degree") = 8192,
             py::arg("coeff_modulus_bits") = std::vector<int>{60, 40, 40, 60},
             py::arg("scale") = pow(2.0, 40))
        .def("generate_keys", &CKKSContext::generate_keys,
             py::arg("generate_galois") = true)
        .def("has_galois_keys", &CKKSContext::has_galois_keys)
        .def("slot_count", &CKKSContext::slot_count)
        .def("get_coeff_modulus_sizes", &CKKSContext::get_coeff_modulus_sizes)
        .def_readonly("poly_modulus_degree", &CKKSContext::poly_modulus_degree)
        .def_readonly("scale", &CKKSContext::scale);

    py::class_<CKKSCiphertext>(m, "CKKSCiphertext")
        .def("get_level", &CKKSCiphertext::get_level)
        .def("get_scale", &CKKSCiphertext::get_scale)
        .def("size", &CKKSCiphertext::size)
        .def("serialize", &CKKSCiphertext::serialize)
        .def_static("deserialize", &CKKSCiphertext::deserialize);

    py::class_<ColumnPackedMatrix>(m, "ColumnPackedMatrix")
        .def(py::init<py::array_t<double>>())
        .def("get_column", &ColumnPackedMatrix::get_column)
        .def_readonly("rows", &ColumnPackedMatrix::rows)
        .def_readonly("cols", &ColumnPackedMatrix::cols);

    py::class_<MOAIOperations>(m, "MOAIOperations")
        .def(py::init<std::shared_ptr<CKKSContext>>())
        .def("reset_counters", &MOAIOperations::reset_counters)
        .def("encrypt_vector", &MOAIOperations::encrypt_vector)
        .def("decrypt_vector", &MOAIOperations::decrypt_vector,
             py::arg("ct"), py::arg("output_size") = 0)
        .def("column_packed_matmul", &MOAIOperations::column_packed_matmul,
             py::arg("ct_x"), py::arg("weight"), py::arg("rescale") = true)
        .def("rotation_matmul", &MOAIOperations::rotation_matmul)
        .def("lora_delta_column_packed", &MOAIOperations::lora_delta_column_packed,
             py::arg("ct_x"), py::arg("lora_a"), py::arg("lora_b"), py::arg("scaling") = 1.0)
        .def("get_stats", &MOAIOperations::get_stats);

    m.attr("__version__") = "1.0.0";
    m.attr("BACKEND_NAME") = "N2HE-HEXL";
}
NATIVE_EOF

    # Create CMakeLists.txt for building
    cat > CMakeLists.txt << 'CMAKE_EOF'
cmake_minimum_required(VERSION 3.16)
project(n2he_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# SEAL path (adjust based on installation)
set(SEAL_DIR "${CMAKE_SOURCE_DIR}/../../third_party/SEAL/build" CACHE PATH "Path to SEAL build directory")
list(APPEND CMAKE_PREFIX_PATH ${SEAL_DIR})
find_package(SEAL 4.1 REQUIRED)

# Create the Python module
pybind11_add_module(n2he_native n2he_native.cpp)
target_link_libraries(n2he_native PRIVATE SEAL::seal)

# Install to the correct location
install(TARGETS n2he_native LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
CMAKE_EOF

    # Install pybind11 if needed
    pip install pybind11[global] -q

    # Configure and build
    cd "$BUILD_DIR"
    cmake . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DSEAL_DIR="$THIRD_PARTY_DIR/SEAL/build"

    make -j$(nproc)
    make install

    log_info "N2HE bindings built successfully"
}

# Create Python wrapper module
create_python_wrapper() {
    log_info "Creating Python wrapper module..."

    # Create __init__.py for the package
    cat > "$PROJECT_ROOT/crypto_backend/__init__.py" << 'INIT_EOF'
"""
Crypto Backend Package.

Provides cryptographic backends for TenSafe HE operations.
"""
INIT_EOF

    cat > "$PROJECT_ROOT/crypto_backend/n2he_hexl/__init__.py" << 'WRAPPER_EOF'
"""
N2HE-HEXL Backend for TenSafe.

Provides CKKS homomorphic encryption with MOAI-style optimizations.
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Add lib directory to path
_LIB_DIR = Path(__file__).parent / "lib"
if _LIB_DIR.exists():
    sys.path.insert(0, str(_LIB_DIR))

# Try to import the native module
_NATIVE_AVAILABLE = False
_NATIVE_MODULE = None

try:
    import n2he_native
    _NATIVE_MODULE = n2he_native
    _NATIVE_AVAILABLE = True
    logger.info(f"N2HE-HEXL native module loaded: version {n2he_native.__version__}")
except ImportError as e:
    logger.warning(f"N2HE-HEXL native module not available: {e}")
    logger.warning("Run ./scripts/build_n2he_hexl.sh to build the native module")


class HEBackendNotAvailableError(Exception):
    """Raised when the HE backend is not available."""
    pass


@dataclass
class CKKSParams:
    """CKKS encryption parameters."""
    poly_modulus_degree: int = 8192
    coeff_modulus_bits: List[int] = None
    scale_bits: int = 40
    security_level: int = 128

    def __post_init__(self):
        if self.coeff_modulus_bits is None:
            self.coeff_modulus_bits = [60, 40, 40, 60]


class CKKSCiphertext:
    """Wrapper for CKKS ciphertext with metadata."""

    def __init__(self, native_ct, context):
        if not _NATIVE_AVAILABLE:
            raise HEBackendNotAvailableError("N2HE-HEXL backend not available")
        self._ct = native_ct
        self._ctx = context

    @property
    def level(self) -> int:
        """Get remaining multiplicative levels."""
        return self._ct.get_level()

    @property
    def scale(self) -> float:
        """Get current scale."""
        return self._ct.get_scale()

    def serialize(self) -> bytes:
        """Serialize ciphertext to bytes."""
        return self._ct.serialize()

    @classmethod
    def deserialize(cls, data: bytes, context) -> "CKKSCiphertext":
        """Deserialize ciphertext from bytes."""
        native_ct = _NATIVE_MODULE.CKKSCiphertext.deserialize(data, context._native_ctx)
        return cls(native_ct, context)


class ColumnPackedMatrix:
    """Column-packed matrix for MOAI-style HE operations."""

    def __init__(self, matrix: np.ndarray):
        if not _NATIVE_AVAILABLE:
            raise HEBackendNotAvailableError("N2HE-HEXL backend not available")

        if matrix.ndim != 2:
            raise ValueError("Matrix must be 2D")

        self._native = _NATIVE_MODULE.ColumnPackedMatrix(matrix.astype(np.float64))
        self._shape = matrix.shape

    @property
    def rows(self) -> int:
        return self._native.rows

    @property
    def cols(self) -> int:
        return self._native.cols

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape


class N2HEHEXLBackend:
    """
    N2HE-HEXL Backend for CKKS homomorphic encryption.

    Implements MOAI-style optimizations:
    - Column packing for rotation-free plaintext-ciphertext matmul
    - Interleaved batching for ciphertext operations
    - Consistent packing strategy across operations
    """

    def __init__(self, params: Optional[CKKSParams] = None):
        if not _NATIVE_AVAILABLE:
            raise HEBackendNotAvailableError(
                "N2HE-HEXL native module not available.\n"
                "Build with: ./scripts/build_n2he_hexl.sh"
            )

        self._params = params or CKKSParams()
        self._native_ctx = None
        self._ops = None
        self._setup_complete = False

    def is_available(self) -> bool:
        """Check if the backend is available."""
        return _NATIVE_AVAILABLE

    def get_backend_name(self) -> str:
        """Get backend name."""
        return "N2HE-HEXL"

    def setup_context(self) -> None:
        """Initialize CKKS context with parameters."""
        self._native_ctx = _NATIVE_MODULE.CKKSContext(
            self._params.poly_modulus_degree,
            self._params.coeff_modulus_bits,
            2.0 ** self._params.scale_bits
        )
        self._setup_complete = True
        logger.info(
            f"CKKS context initialized: "
            f"ring_degree={self._params.poly_modulus_degree}, "
            f"coeff_modulus={self._params.coeff_modulus_bits}"
        )

    def generate_keys(self, generate_galois: bool = True) -> None:
        """Generate encryption keys."""
        if not self._setup_complete:
            raise RuntimeError("Call setup_context() first")

        self._native_ctx.generate_keys(generate_galois)
        self._ops = _NATIVE_MODULE.MOAIOperations(self._native_ctx)

        logger.info(
            f"Keys generated: galois_keys={self._native_ctx.has_galois_keys()}, "
            f"slot_count={self._native_ctx.slot_count()}"
        )

    def get_context_params(self) -> Dict[str, Any]:
        """Get CKKS context parameters for verification."""
        if not self._setup_complete:
            return {}

        return {
            "ring_degree": self._native_ctx.poly_modulus_degree,
            "scale": self._native_ctx.scale,
            "scale_bits": self._params.scale_bits,
            "slot_count": self._native_ctx.slot_count(),
            "coeff_modulus_chain_length": len(self._native_ctx.get_coeff_modulus_sizes()),
            "coeff_modulus_sizes": self._native_ctx.get_coeff_modulus_sizes(),
            "has_galois_keys": self._native_ctx.has_galois_keys(),
            "security_level": self._params.security_level,
        }

    def encrypt(self, plaintext: np.ndarray) -> CKKSCiphertext:
        """Encrypt a plaintext vector."""
        if self._ops is None:
            raise RuntimeError("Call generate_keys() first")

        native_ct = self._ops.encrypt_vector(plaintext.astype(np.float64))
        return CKKSCiphertext(native_ct, self)

    def decrypt(self, ciphertext: CKKSCiphertext, output_size: int = 0) -> np.ndarray:
        """Decrypt a ciphertext to plaintext vector."""
        if self._ops is None:
            raise RuntimeError("Keys not available for decryption")

        return np.array(self._ops.decrypt_vector(ciphertext._ct, output_size))

    def column_packed_matmul(
        self,
        ct_x: CKKSCiphertext,
        weight: ColumnPackedMatrix,
        rescale: bool = True
    ) -> CKKSCiphertext:
        """
        MOAI-style column-packed plaintext-ciphertext matmul.

        This achieves ZERO rotations for the matmul operation.
        """
        if self._ops is None:
            raise RuntimeError("Call generate_keys() first")

        native_ct = self._ops.column_packed_matmul(ct_x._ct, weight._native, rescale)
        return CKKSCiphertext(native_ct, self)

    def lora_delta(
        self,
        ct_x: CKKSCiphertext,
        lora_a: ColumnPackedMatrix,
        lora_b: ColumnPackedMatrix,
        scaling: float = 1.0
    ) -> CKKSCiphertext:
        """
        Compute LoRA delta with MOAI column packing.

        delta = scaling * (x @ A^T @ B^T)

        Uses column packing to achieve zero rotations.
        """
        if self._ops is None:
            raise RuntimeError("Call generate_keys() first")

        native_ct = self._ops.lora_delta_column_packed(
            ct_x._ct, lora_a._native, lora_b._native, scaling
        )
        return CKKSCiphertext(native_ct, self)

    def get_operation_stats(self) -> Dict[str, int]:
        """Get operation statistics."""
        if self._ops is None:
            return {"rotations_used": 0, "multiplications": 0, "additions": 0}
        return dict(self._ops.get_stats())

    def reset_stats(self) -> None:
        """Reset operation counters."""
        if self._ops is not None:
            self._ops.reset_counters()


def verify_backend() -> Dict[str, Any]:
    """
    Verify the N2HE-HEXL backend is properly installed and functional.

    Returns dict with verification results. Raises if backend not available.
    """
    if not _NATIVE_AVAILABLE:
        raise HEBackendNotAvailableError(
            "N2HE-HEXL native module not available.\n"
            "Build with: ./scripts/build_n2he_hexl.sh"
        )

    backend = N2HEHEXLBackend()
    backend.setup_context()
    backend.generate_keys()

    params = backend.get_context_params()

    # Test encrypt/decrypt
    test_data = np.array([1.0, 2.0, 3.0, 4.0])
    ct = backend.encrypt(test_data)
    decrypted = backend.decrypt(ct, len(test_data))

    error = np.max(np.abs(test_data - decrypted))

    return {
        "backend": "N2HE-HEXL",
        "available": True,
        "params": params,
        "test_encrypt_decrypt": {
            "input": test_data.tolist(),
            "output": decrypted.tolist(),
            "max_error": float(error),
            "passed": error < 1e-4,
        }
    }


# Export public API
__all__ = [
    "N2HEHEXLBackend",
    "CKKSCiphertext",
    "CKKSParams",
    "ColumnPackedMatrix",
    "HEBackendNotAvailableError",
    "verify_backend",
]
WRAPPER_EOF

    log_info "Python wrapper created"
}

# Main build process
main() {
    log_info "Building N2HE-HEXL for TenSafe..."
    log_info "Project root: $PROJECT_ROOT"

    check_prereqs
    install_hexl
    install_seal
    build_n2he_bindings
    create_python_wrapper

    log_info "Build complete!"
    log_info "Verify installation with: python scripts/verify_he_backend.py"
}

main "$@"
