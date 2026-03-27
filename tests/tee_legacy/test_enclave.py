"""
Unit Tests for Core Enclave Infrastructure
===========================================

Tests for:
- Enclave creation and management
- EnclaveSession lifecycle
- SecureMemory allocation
- Attestation simulation
- Sealed storage
"""

import pytest
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tee_ml.core.enclave import (
    Enclave,
    EnclaveSession,
    SecureMemory,
    EnclaveError,
    EnclaveMemoryError,
    EnclaveSecurityError,
    create_enclave,
)
from tee_ml.core.attestation import (
    AttestationReport,
    AttestationService,
    AttestationResult,
    simulate_remote_attestation,
    create_mock_ias,
)
from tee_ml.core.sealed_storage import (
    SealedStorage,
    SealedData,
    SealedStorageError,
    seal_model_weights,
    unseal_model_weights,
    create_sealed_storage,
)
from tee_ml.simulation.overhead import (
    OverheadModel,
    OverheadSimulator,
    estimate_inference_overhead,
    compare_tee_vs_he,
)


class TestSecureMemory:
    """Test SecureMemory isolation and allocation."""

    @pytest.fixture
    def secure_memory(self):
        return SecureMemory(size_bytes=1024 * 1024)  # 1 MB

    def test_allocate_and_read(self, secure_memory):
        """Test allocating and reading data."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        offset = secure_memory.allocate(data)

        read_data = secure_memory.read(offset)
        assert np.array_equal(read_data, data)

    def test_allocate_multiple(self, secure_memory):
        """Test allocating multiple data blocks."""
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([4.0, 5.0, 6.0])

        offset1 = secure_memory.allocate(data1)
        offset2 = secure_memory.allocate(data2)

        assert offset1 != offset2

        read_data1 = secure_memory.read(offset1)
        read_data2 = secure_memory.read(offset2)

        assert np.array_equal(read_data1, data1)
        assert np.array_equal(read_data2, data2)

    def test_free_memory(self, secure_memory):
        """Test freeing allocated memory."""
        data = np.array([1.0, 2.0, 3.0])
        offset = secure_memory.allocate(data)

        initial_usage = secure_memory.get_usage()
        assert initial_usage > 0

        secure_memory.free(offset)

        # Usage should decrease
        assert secure_memory.get_usage() < initial_usage

    def test_memory_limit(self, secure_memory):
        """Test that memory limit is enforced."""
        # Allocate near limit (900 KB of actual data)
        # np.zeros with integer gives int64 which is 8 bytes per element
        large_data = np.zeros(int(900 * 1024 / 8), dtype=np.int64)  # ~900 KB
        secure_memory.allocate(large_data)

        # Try to allocate beyond limit
        too_large = np.zeros(int(200 * 1024 / 8), dtype=np.int64)  # ~200 KB

        with pytest.raises(EnclaveMemoryError):
            secure_memory.allocate(too_large)

    def test_memory_utilization(self, secure_memory):
        """Test memory utilization calculation."""
        assert secure_memory.get_utilization() == 0.0

        # Allocate 512 KB of actual data
        data = np.zeros(int(512 * 1024 / 8), dtype=np.int64)  # 512 KB
        secure_memory.allocate(data)

        # Should be ~50% utilized
        utilization = secure_memory.get_utilization()
        assert 0.4 < utilization < 0.6

    def test_invalid_offset(self, secure_memory):
        """Test reading from invalid offset."""
        with pytest.raises(KeyError):
            secure_memory.read(999)


class TestEnclave:
    """Test Enclave creation and management."""

    @pytest.fixture
    def enclave(self):
        return Enclave(enclave_id="test-enclave", memory_limit_mb=128)

    def test_enclave_creation(self, enclave):
        """Test creating an enclave."""
        assert enclave.enclave_id == "test-enclave"
        assert enclave.memory_limit_bytes == 128 * 1024 * 1024
        assert enclave.is_isolated()

    def test_enclave_enter_exit(self, enclave):
        """Test entering and exiting enclave."""
        data = np.array([1.0, 2.0, 3.0, 4.0])

        session = enclave.enter(data)
        assert session.is_active()

        result = enclave.exit(session)
        assert not session.is_active()
        assert np.array_equal(result, data)

    def test_enclave_session_execute(self, enclave):
        """Test executing operation in enclave."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        session = enclave.enter(data)

        def double(x):
            return x * 2

        result = session.execute(double)
        assert np.array_equal(result, np.array([2.0, 4.0, 6.0, 8.0]))

    def test_enclave_session_inactive_execute(self, enclave):
        """Test that inactive session cannot execute."""
        data = np.array([1.0, 2.0, 3.0])
        session = enclave.enter(data)
        enclave.exit(session)

        with pytest.raises(EnclaveSecurityError):
            session.execute(lambda x: x * 2)

    def test_enclave_statistics(self, enclave):
        """Test enclave statistics."""
        data = np.array([1.0, 2.0, 3.0])
        session = enclave.enter(data)

        stats = enclave.get_statistics()
        assert stats["active_sessions"] == 1
        assert stats["total_entries"] == 1
        assert stats["total_exits"] == 0

        enclave.exit(session)

        stats = enclave.get_statistics()
        assert stats["active_sessions"] == 0
        assert stats["total_entries"] == 1
        assert stats["total_exits"] == 1

    def test_enclave_memory_exceeds_limit(self, enclave):
        """Test that large data is rejected."""
        # Create data larger than enclave memory
        large_data = np.zeros(200 * 1024 * 1024, dtype=np.float32)  # 200 MB

        with pytest.raises(EnclaveMemoryError):
            enclave.enter(large_data)

    def test_enclave_get_measurement(self, enclave):
        """Test getting enclave measurement."""
        measurement = enclave.get_measurement()
        assert isinstance(measurement, bytes)
        assert len(measurement) == 32  # SHA256

    def test_create_enclave_factory(self):
        """Test create_enclave factory function."""
        enclave = create_enclave(memory_limit_mb=64)
        assert enclave.memory_limit_bytes == 64 * 1024 * 1024


class TestEnclaveSession:
    """Test EnclaveSession lifecycle."""

    @pytest.fixture
    def session(self):
        enclave = Enclave(enclave_id="test-enclave")
        data = np.array([1.0, 2.0, 3.0])
        return enclave.enter(data)

    def test_session_id_unique(self):
        """Test that sessions have unique IDs."""
        enclave = Enclave(enclave_id="test-enclave")
        data = np.array([1.0, 2.0, 3.0])

        session1 = enclave.enter(data)
        session2 = enclave.enter(data)

        assert session1.session_id != session2.session_id

    def test_session_get_memory_usage(self, session):
        """Test getting session memory usage."""
        usage = session.get_memory_usage()
        assert usage > 0

    def test_session_duration(self, session):
        """Test session duration tracking."""
        # Session should have positive duration
        duration = session.get_session_duration_ns()
        assert duration >= 0

        # Close session and check duration
        session.close()
        final_duration = session.get_session_duration_ns()
        assert final_duration >= duration


class TestAttestation:
    """Test attestation simulation."""

    @pytest.fixture
    def enclave(self):
        return Enclave(enclave_id="test-enclave")

    @pytest.fixture
    def attestation_service(self):
        return AttestationService()

    def test_generate_report(self, attestation_service, enclave):
        """Test generating attestation report."""
        nonce = attestation_service.generate_nonce()
        report = attestation_service.generate_report(enclave, nonce=nonce)

        assert report.enclave_id == enclave.enclave_id
        assert report.measurement == enclave.get_measurement()
        assert report.nonce == nonce
        assert len(report.signature) > 0

    def test_verify_report_valid(self, attestation_service, enclave):
        """Test verifying valid report."""
        # Register enclave
        attestation_service.register_enclave(
            enclave.enclave_id,
            enclave.get_measurement()
        )

        # Generate and verify report
        nonce = attestation_service.generate_nonce()
        report = attestation_service.generate_report(enclave, nonce=nonce)
        result = attestation_service.verify_report(report)

        assert result.valid
        assert result.measurement_match
        assert result.signature_valid
        assert result.timestamp_recent

    def test_verify_report_measurement_mismatch(self, attestation_service, enclave):
        """Test verification with wrong measurement."""
        # Register wrong measurement
        wrong_measurement = b"x" * 32
        attestation_service.register_enclave(enclave.enclave_id, wrong_measurement)

        # Generate report with correct measurement
        nonce = attestation_service.generate_nonce()
        report = attestation_service.generate_report(enclave, nonce=nonce)

        # Verification should fail
        result = attestation_service.verify_report(report)
        assert not result.valid
        assert not result.measurement_match

    def test_report_serialization(self, attestation_service, enclave):
        """Test report JSON serialization."""
        nonce = attestation_service.generate_nonce()
        report = attestation_service.generate_report(enclave, nonce=nonce)

        # Convert to JSON and back
        json_str = report.to_json()
        restored = AttestationReport.from_json(json_str)

        assert restored.enclave_id == report.enclave_id
        assert restored.measurement == report.measurement
        assert restored.nonce == report.nonce

    def test_remote_attestation(self, enclave):
        """Test complete remote attestation flow."""
        result = simulate_remote_attestation(enclave, "challenger")

        assert result["result"]["valid"]
        assert result["enclave_id"] == enclave.enclave_id
        assert "nonce" in result

    def test_generate_nonce(self, attestation_service):
        """Test nonce generation."""
        nonce1 = attestation_service.generate_nonce()
        nonce2 = attestation_service.generate_nonce()

        assert len(nonce1) == 16
        assert len(nonce2) == 16
        assert nonce1 != nonce2

    def test_attestation_history(self, attestation_service, enclave):
        """Test attestation history tracking."""
        initial_history = attestation_service.get_attestation_history()
        initial_count = len(initial_history)

        # Generate some reports
        for _ in range(3):
            nonce = attestation_service.generate_nonce()
            attestation_service.generate_report(enclave, nonce=nonce)

        history = attestation_service.get_attestation_history()
        assert len(history) == initial_count + 3


class TestSealedStorage:
    """Test sealed storage functionality."""

    @pytest.fixture
    def enclave(self):
        return Enclave(enclave_id="test-enclave")

    @pytest.fixture
    def storage(self, tmp_path):
        return SealedStorage(storage_path=str(tmp_path))

    def test_seal_unseal(self, storage, enclave):
        """Test sealing and unsealing data."""
        data = b"secret data"

        sealed = storage.seal(
            data,
            enclave.enclave_id,
            enclave.get_measurement()
        )

        unsealed = storage.unseal(
            sealed,
            enclave.enclave_id,
            enclave.get_measurement()
        )

        assert unsealed == data

    def test_seal_wrong_enclave(self, storage, enclave):
        """Test that wrong enclave cannot unseal."""
        data = b"secret data"
        sealed = storage.seal(
            data,
            enclave.enclave_id,
            enclave.get_measurement()
        )

        # Try with different measurement
        wrong_measurement = b"x" * 32
        with pytest.raises(SealedStorageError):
            storage.unseal(
                sealed,
                enclave.enclave_id,
                wrong_measurement
            )

    def test_save_load_sealed(self, storage, enclave):
        """Test saving and loading sealed data."""
        data = b"persistent secret"

        storage.save_sealed(
            "test_key",
            data,
            enclave.enclave_id,
            enclave.get_measurement()
        )

        loaded = storage.load_sealed(
            "test_key",
            enclave.enclave_id,
            enclave.get_measurement()
        )

        assert loaded == data

    def test_sealed_data_serialization(self, storage, enclave):
        """Test sealed data JSON serialization."""
        data = b"secret"
        sealed = storage.seal(
            data,
            enclave.enclave_id,
            enclave.get_measurement()
        )

        json_str = sealed.to_json()
        restored = SealedData.from_json(json_str)

        assert restored.ciphertext == sealed.ciphertext
        assert restored.enclave_id == sealed.enclave_id

    def test_list_sealed_keys(self, storage, enclave):
        """Test listing sealed storage keys."""
        # Save multiple sealed items
        for i in range(3):
            data = f"secret{i}".encode()
            storage.save_sealed(
                f"key{i}",
                data,
                enclave.enclave_id,
                enclave.get_measurement()
            )

        keys = storage.list_sealed_keys()
        assert len(keys) == 3
        assert "key0" in keys
        assert "key1" in keys
        assert "key2" in keys

    def test_delete_sealed(self, storage, enclave):
        """Test deleting sealed data."""
        data = b"secret"
        storage.save_sealed(
            "test_key",
            data,
            enclave.enclave_id,
            enclave.get_measurement()
        )

        assert "test_key" in storage.list_sealed_keys()

        storage.delete_sealed("test_key")

        assert "test_key" not in storage.list_sealed_keys()

    def test_seal_model_weights(self, enclave, tmp_path):
        """Test sealing model weights."""
        weights = {
            "layer1": b"weights1",
            "layer2": b"weights2",
        }

        storage = SealedStorage(storage_path=str(tmp_path))
        keys = seal_model_weights(weights, enclave, storage)

        assert "layer1" in keys
        assert "layer2" in keys

        # Unseal and verify
        unsealed = unseal_model_weights(keys, enclave, storage)
        assert unsealed == weights


class TestOverheadModel:
    """Test overhead model."""

    def test_calculate_overhead(self):
        """Test overhead calculation."""
        model = OverheadModel()

        overhead = model.calculate_overhead(
            operation_time_ns=1000,  # 1 μs
            data_size_mb=1.0,
            num_entries=1,
            num_exits=1,
        )

        assert overhead["entry_overhead_ns"] == model.entry_ns
        assert overhead["exit_overhead_ns"] == model.exit_ns
        assert overhead["memory_encryption_ns"] > 0
        assert overhead["total_overhead_ns"] > 0
        assert overhead["slowdown_factor"] > 1.0

    def test_custom_overhead_model(self):
        """Test custom overhead parameters."""
        model = OverheadModel(
            entry_ns=10000,
            exit_ns=5000,
            memory_encryption_ns_per_mb=200,
        )

        overhead = model.calculate_overhead(
            operation_time_ns=1000,
            data_size_mb=2.0,
        )

        assert overhead["entry_overhead_ns"] == 10000
        assert overhead["exit_overhead_ns"] == 5000
        assert overhead["memory_encryption_ns"] == 400  # 200 * 2

    def test_estimate_total_time(self):
        """Test estimating total time."""
        model = OverheadModel()

        total = model.estimate_total_time(
            plaintext_time_ns=1000,
            data_size_mb=1.0,
        )

        assert total > 1000  # Should be more than plaintext time

    def test_calculate_epc_overhead(self):
        """Test EPC overhead calculation."""
        model = OverheadModel()

        # Fits in EPC
        epc_ok = model.calculate_epc_overhead(memory_mb=64, epc_size_mb=128)
        assert epc_ok["fits_in_epc"]
        assert epc_ok["paging_overhead_ns"] == 0

        # Exceeds EPC
        epc_exceed = model.calculate_epc_overhead(memory_mb=256, epc_size_mb=128)
        assert not epc_exceed["fits_in_epc"]
        assert epc_exceed["paging_overhead_ns"] > 0


class TestOverheadSimulator:
    """Test overhead simulator."""

    @pytest.fixture
    def simulator(self):
        return OverheadSimulator()

    def test_simulate_operation(self, simulator):
        """Test simulating an operation."""
        data = np.array([1.0, 2.0, 3.0, 4.0])

        def simple_op(x):
            return x * 2

        result, metrics = simulator.simulate_operation(simple_op, data)

        assert np.array_equal(result, np.array([2.0, 4.0, 6.0, 8.0]))
        assert metrics.operation_name == "simple_op"
        assert metrics.slowdown_factor > 1.0

    def test_get_average_slowdown(self, simulator):
        """Test getting average slowdown."""
        data = np.array([1.0, 2.0, 3.0])

        for _ in range(5):
            simulator.simulate_operation(lambda x: x * 2, data)

        avg_slowdown = simulator.get_average_slowdown()
        assert avg_slowdown > 1.0

    def test_reset_metrics(self, simulator):
        """Test resetting metrics."""
        data = np.array([1.0, 2.0, 3.0])
        simulator.simulate_operation(lambda x: x * 2, data)

        assert len(simulator.metrics_history) > 0

        simulator.reset_metrics()
        assert len(simulator.metrics_history) == 0

    def test_get_metrics_summary(self, simulator):
        """Test getting metrics summary."""
        data = np.array([1.0, 2.0, 3.0])

        for _ in range(3):
            simulator.simulate_operation(lambda x: x * 2, data)

        summary = simulator.get_metrics_summary()
        assert summary["num_operations"] == 3
        assert summary["avg_slowdown"] > 1.0


class TestIntegration:
    """Integration tests for core infrastructure."""

    def test_complete_enclave_workflow(self):
        """Test complete enclave workflow."""
        # Create enclave
        enclave = Enclave(enclave_id="workflow-enclave")

        # Enter with data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        session = enclave.enter(data)

        # Execute multiple operations
        result = session.execute(lambda x: x * 2)
        result = session.execute(lambda x: x + 10)

        # Exit enclave
        final = enclave.exit(session)

        # Verify
        assert np.array_equal(final, data)

    def test_attestation_with_enclave(self):
        """Test attestation with real enclave."""
        enclave = Enclave(enclave_id="attest-enclave")
        service = AttestationService()

        # Register enclave
        service.register_enclave(enclave.enclave_id, enclave.get_measurement())

        # Attest
        nonce = service.generate_nonce()
        report = service.generate_report(enclave, nonce=nonce)
        result = service.verify_report(report)

        assert result.valid

    def test_sealed_storage_with_enclave(self, tmp_path):
        """Test sealed storage with real enclave."""
        enclave = Enclave(enclave_id="storage-enclave")
        storage = SealedStorage(storage_path=str(tmp_path))

        # Seal some data
        original_data = b"model weights v1.0"
        storage.save_sealed(
            "model",
            original_data,
            enclave.enclave_id,
            enclave.get_measurement(),
        )

        # Unseal
        loaded_data = storage.load_sealed(
            "model",
            enclave.enclave_id,
            enclave.get_measurement(),
        )

        assert loaded_data == original_data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
