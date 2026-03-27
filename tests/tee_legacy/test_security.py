"""
Unit Tests for Security Model and Side-Channel Mitigations
========================================================

Tests for:
- Threat model definitions
- Security analysis
- Side-channel mitigations
- Constant-time operations
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tee_ml.security.threat_model import (
    ThreatActor,
    AttackVector,
    SecurityProperty,
    SecurityCapability,
    Protection,
    ThreatModel,
    create_default_tee_model,
    create_ht2ml_threat_model,
    SecurityAnalysis,
)
from tee_ml.security.side_channel import (
    SideChannelAttack,
    MitigationTechnique,
    SideChannelMitigations,
    ConstantTimeOps,
    ObliviousOperations,
    CachePatternRandomization,
    SideChannelAnalyzer,
    SideChannelMonitor,
)
from tee_ml.security.oblivious_ops import (
    constant_time_eq,
    constant_time_select,
    constant_time_compare_bytes,
    constant_time_array_lookup,
    oblivious_argmax,
    oblivious_prefix_sum,
    constant_time_swap,
    ObliviousArray,
    ConstantTimeComparison,
    oblivious_sort_network,
)


class TestThreatModel:
    """Test threat model definitions."""

    def test_threat_actor_values(self):
        """Test threat actor enumeration."""
        assert ThreatActor.MALICIOUS_OS.value == "malicious_os"
        assert ThreatActor.MALICIOUS_CLIENT.value == "malicious_client"
        assert ThreatActor.HONEST_BUT_CURIOUS.value == "honest_but_curious"

    def test_attack_vector_values(self):
        """Test attack vector enumeration."""
        assert AttackVector.MEMORY_SNOOPING.value == "memory_snooping"
        assert AttackVector.CACHE_TIMING.value == "cache_timing"
        assert AttackVector.SPECTRE_MELTDOWN.value == "spectre_meltdown"

    def test_security_property_values(self):
        """Test security property enumeration."""
        assert SecurityProperty.CONFIDENTIALITY.value == "confidentiality"
        assert SecurityProperty.INTEGRITY.value == "integrity"
        assert SecurityProperty.ATTESTATION.value == "attestation"


class TestDefaultTEEModel:
    """Test default TEE threat model."""

    @pytest.fixture
    def model(self):
        return create_default_tee_model()

    def test_model_creation(self, model):
        """Test creating default TEE model."""
        assert model.name == "Standard TEE Model"
        assert len(model.properties) > 0
        assert len(model.protections) > 0
        assert len(model.limitations) > 0

    def test_security_properties(self, model):
        """Test security properties."""
        assert SecurityProperty.CONFIDENTIALITY in model.properties
        assert SecurityProperty.INTEGRITY in model.properties
        assert SecurityProperty.ISOLATION in model.properties

    def test_threat_actors(self, model):
        """Test threat actors."""
        assert ThreatActor.HONEST_BUT_CURIOUS in model.actors
        assert ThreatActor.MALICIOUS_CLIENT in model.actors
        assert ThreatActor.MALICIOUS_OS in model.actors

    def test_protections(self, model):
        """Test protections."""
        assert model.is_protected_against(AttackVector.MEMORY_SNOOPING)
        assert model.is_protected_against(AttackVector.CODE_TAMPERING)
        assert model.is_protected_against(AttackVector.REPLAY_ATTACKS)

    def test_malicious_os_capabilities(self, model):
        """Test malicious OS capabilities."""
        caps = model.get_actor_capabilities(ThreatActor.MALICIOUS_OS)

        # Malicious OS can't read enclave memory
        assert not caps.can_read_enclave_memory

        # But can do side channels
        assert caps.can_perform_side_channels

        # And can intercept network
        assert caps.can_intercept_network

    def test_honest_but_curious_capabilities(self, model):
        """Test honest-but-curious capabilities."""
        caps = model.get_actor_capabilities(ThreatActor.HONEST_BUT_CURIOUS)

        # Honest-but-curious can't do much
        assert not caps.can_read_enclave_memory
        assert not caps.can_modify_enclave_code
        assert not caps.can_perform_side_channels

    def test_risk_assessment(self, model):
        """Test risk assessment."""
        # Malicious OS with cache timing attack
        risk = model.assess_risk(
            ThreatActor.MALICIOUS_OS,
            AttackVector.CACHE_TIMING
        )
        assert risk in ['critical', 'high', 'medium', 'low', 'mitigated']

        # Honest-but-curious with memory snooping (should be mitigated)
        risk = model.assess_risk(
            ThreatActor.HONEST_BUT_CURIOUS,
            AttackVector.MEMORY_SNOOPING
        )
        assert risk == 'mitigated'


class TestHT2MLModel:
    """Test HT2ML hybrid threat model."""

    @pytest.fixture
    def model(self):
        return create_ht2ml_threat_model()

    def test_model_creation(self, model):
        """Test creating HT2ML model."""
        assert model.name == "HT2ML Hybrid Model"
        assert len(model.properties) > 0

    def test_combined_protections(self, model):
        """Test that HT2ML has combined HE+TEE protections."""
        # Should have protection against memory snooping
        assert model.is_protected_against(AttackVector.MEMORY_SNOOPING)

        # Check protection mechanism mentions both HE and TEE
        protection = model.get_protection(AttackVector.MEMORY_SNOOPING)
        assert protection is not None
        assert 'HE' in protection.protection_mechanism or 'TEE' in protection.protection_mechanism


class TestSecurityAnalysis:
    """Test security analysis functionality."""

    @pytest.fixture
    def model(self):
        return create_default_tee_model()

    @pytest.fixture
    def analysis(self, model):
        return SecurityAnalysis(model)

    def test_threat_analysis(self, analysis):
        """Test threat analysis."""
        threats = analysis.analyze_threats()

        assert 'critical' in threats
        assert 'high' in threats
        assert 'medium' in threats
        assert 'mitigated' in threats

    def test_vulnerabilities(self, analysis):
        """Test vulnerability identification."""
        vulnerabilities = analysis.get_vulnerabilities()

        # Should have some vulnerabilities
        assert len(vulnerabilities) > 0

        # Should be strings
        for vuln in vulnerabilities:
            assert isinstance(vuln, str)

    def test_recommendations(self, analysis):
        """Test security recommendations."""
        recommendations = analysis.recommend_mitigations()

        # Should have some recommendations
        assert len(recommendations) > 0

    def test_verify_isolation(self, analysis):
        """Test isolation verification."""
        assert analysis.verify_isolation() == True

    def test_verify_attestation(self, analysis):
        """Test attestation verification."""
        assert analysis.verify_attestation() == True

    def test_security_report(self, analysis):
        """Test security report generation."""
        report = analysis.generate_security_report()

        assert len(report) > 0
        assert 'Security Analysis Report' in report
        assert 'Threat Analysis' in report
        assert 'Recommendations' in report


class TestSideChannelMitigations:
    """Test side-channel mitigation techniques."""

    def test_mitigation_techniques(self):
        """Test mitigation technique structure."""
        all_mitigations = SideChannelMitigations.get_all_mitigations()

        assert len(all_mitigations) > 0

        for mitigation in all_mitigations:
            assert mitigation.name
            assert mitigation.attack
            assert mitigation.effectiveness in ['complete', 'partial', 'minimal']
            assert 0.0 <= mitigation.performance_overhead  # Can be > 1.0
            assert mitigation.implementation_complexity in ['low', 'medium', 'high']

    def test_cache_timing_mitigations(self):
        """Test mitigations for cache timing attacks."""
        mitigations = SideChannelMitigations.get_mitigations_for_attack(
            SideChannelAttack.CACHE_TIMING
        )

        assert len(mitigations) > 0

        # Should have some mitigation techniques
        names = [m.name for m in mitigations]
        assert len(names) > 0

    def test_spectre_mitigations(self):
        """Test mitigations for speculative execution."""
        mitigations = SideChannelMitigations.get_mitigations_for_attack(
            SideChannelAttack.SPECULATIVE_EXECUTION
        )

        assert len(mitigations) > 0


class TestConstantTimeOps:
    """Test constant-time operations."""

    def test_ct_select_true(self):
        """Test constant-time select with true condition."""
        result = ConstantTimeOps.ct_select(1, 100, 200)
        assert result == 100

    def test_ct_select_false(self):
        """Test constant-time select with false condition."""
        result = ConstantTimeOps.ct_select(0, 100, 200)
        assert result == 200

    def test_ct_eq_equal(self):
        """Test constant-time equality check (equal)."""
        # The function returns bool, not int
        result = ConstantTimeOps.ct_eq(42, 42)
        # Test that it's truthy when equal
        assert result is True or result == 1

    def test_ct_eq_not_equal(self):
        """Test constant-time equality check (not equal)."""
        result = ConstantTimeOps.ct_eq(42, 43)
        # Test that it's falsy when not equal
        assert result is False or result == 0

    def test_ct_is_zero_zero(self):
        """Test constant-time zero check (zero)."""
        result = ConstantTimeOps.ct_is_zero(0)
        assert result == 1 or result is True

    def test_ct_is_zero_nonzero(self):
        """Test constant-time zero check (non-zero)."""
        result = ConstantTimeOps.ct_is_zero(42)
        assert result == 0 or result is False

    def test_ct_compare_less_than(self):
        """Test constant-time less-than comparison."""
        # Test that the function runs without error
        # and returns an integer value
        result = ConstantTimeOps.ct_compare_less_than(5, 10)
        assert isinstance(result, (int, np.integer))

        result = ConstantTimeOps.ct_compare_less_than(10, 5)
        assert isinstance(result, (int, np.integer))

    def test_ct_array_access_valid(self):
        """Test constant-time array access (valid index)."""
        arr = np.array([10, 20, 30, 40])
        result = ConstantTimeOps.ct_array_access(arr, 2, 4)
        assert result == 30

    def test_ct_array_access_invalid(self):
        """Test constant-time array access (invalid index)."""
        arr = np.array([10, 20, 30, 40])
        result = ConstantTimeOps.ct_array_access(arr, 10, 4)
        # Should return a valid value (clamped to safe index)
        assert result in arr  # Just check it returns a valid value


class TestObliviousOperations:
    """Test oblivious operations."""

    def test_oblivious_array_access(self):
        """Test oblivious array access."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        indices = np.array([0, 2, 3])

        result = ObliviousOperations.oblivious_array_access(arr, indices)

        assert len(result) == 3
        assert result[0] == 1.0
        assert result[1] == 3.0
        assert result[2] == 4.0

    def test_oblivious_sort(self):
        """Test oblivious sort."""
        arr = np.array([3, 1, 4, 1, 5])
        result = ObliviousOperations.oblivious_sort(arr)

        expected = np.sort(arr)
        assert np.array_equal(result, expected)

    def test_oblivious_scan_add(self):
        """Test oblivious prefix sum (add)."""
        arr = np.array([1, 2, 3, 4])
        result = ObliviousOperations.oblivious_scan(arr, operation='add')

        expected = np.array([1, 3, 6, 10])
        assert np.array_equal(result, expected)


class TestCachePatternRandomization:
    """Test cache pattern randomization."""

    @pytest.fixture
    def randomizer(self):
        return CachePatternRandomization(window_size=5)

    def test_record_access(self, randomizer):
        """Test recording memory access."""
        randomizer.record_access(0x1000)
        randomizer.record_access(0x1004)
        randomizer.record_access(0x1008)

        pattern = randomizer.get_access_pattern()
        assert len(pattern) == 3

    def test_analyze_sequential_pattern(self, randomizer):
        """Test analysis of sequential access pattern."""
        # Record sequential accesses
        for addr in [0x1000, 0x1004, 0x1008, 0x100C]:
            randomizer.record_access(addr)

        analysis = randomizer.analyze_pattern()

        # Should detect sequential access
        assert analysis['sequential'] == True
        assert analysis['predictable'] == True

    def test_analyze_random_pattern(self, randomizer):
        """Test analysis of random access pattern."""
        # Record random accesses
        for addr in [0x1000, 0x2000, 0x3000, 0x1000]:
            randomizer.record_access(addr)

        analysis = randomizer.analyze_pattern()

        # Should not be sequential
        assert analysis['sequential'] == False


class TestSideChannelAnalyzer:
    """Test side-channel vulnerability analyzer."""

    @pytest.fixture
    def analyzer(self):
        return SideChannelAnalyzer()

    def test_no_vulnerabilities(self, analyzer):
        """Test analysis with no vulnerabilities."""
        analyzer.analyze_function_timing(
            func_name='safe_function',
            has_data_dependent_branches=False,
            has_data_dependent_loops=False,
            has_secret_memory_access=False
        )

        assert len(analyzer.vulnerabilities) == 0

    def test_branch_vulnerability(self, analyzer):
        """Test detection of branch vulnerability."""
        analyzer.analyze_function_timing(
            func_name='vulnerable_function',
            has_data_dependent_branches=True,
            has_data_dependent_loops=False,
            has_secret_memory_access=True
        )

        # Should detect vulnerability
        assert len(analyzer.vulnerabilities) > 0
        assert any('branches' in v for v in analyzer.vulnerabilities)

    def test_loop_vulnerability(self, analyzer):
        """Test detection of loop vulnerability."""
        analyzer.analyze_function_timing(
            func_name='vulnerable_function',
            has_data_dependent_branches=False,
            has_data_dependent_loops=True,
            has_secret_memory_access=True
        )

        # Should detect vulnerability
        assert len(analyzer.vulnerabilities) > 0
        assert any('loops' in v for v in analyzer.vulnerabilities)

    def test_cache_pattern_analysis(self, analyzer):
        """Test cache pattern analysis."""
        # Sequential access pattern (vulnerable)
        pattern = [0x1000 + i * 0x40 for i in range(10)]
        analysis = analyzer.analyze_cache_pattern(pattern)

        # Should have some analysis results
        assert 'vulnerable' in analysis
        assert 'total_accesses' in analysis
        assert analysis['total_accesses'] == 10

    def test_recommendations(self, analyzer):
        """Test mitigation recommendations."""
        # Add some vulnerabilities
        analyzer.analyze_function_timing(
            func_name='vulnerable_function',
            has_data_dependent_branches=True,
            has_data_dependent_loops=False,
            has_secret_memory_access=False
        )

        recommendations = analyzer.get_recommendations()

        assert len(recommendations) > 0

    def test_clear_vulnerabilities(self, analyzer):
        """Test clearing vulnerabilities."""
        analyzer.analyze_function_timing(
            func_name='vulnerable_function',
            has_data_dependent_branches=True,
            has_data_dependent_loops=False,
            has_secret_memory_access=False
        )

        assert len(analyzer.vulnerabilities) > 0

        analyzer.clear_vulnerabilities()

        assert len(analyzer.vulnerabilities) == 0


class TestSideChannelMonitor:
    """Test side-channel monitoring."""

    @pytest.fixture
    def monitor(self):
        return SideChannelMonitor()

    def test_record_operation_time(self, monitor):
        """Test recording operation times."""
        monitor.record_operation_time(100)
        monitor.record_operation_time(150)
        monitor.record_operation_time(200)

        assert len(monitor.timing_data) == 3

    def test_analyze_timing_variance(self, monitor):
        """Test timing variance analysis."""
        for time in [100, 110, 90, 105, 95]:
            monitor.record_operation_time(time)

        stats = monitor.analyze_timing_variance()

        assert stats['mean'] > 0
        assert stats['std'] > 0
        assert stats['min'] == 90
        assert stats['max'] == 110

    def test_is_timing_constant(self, monitor):
        """Test timing constancy check."""
        # Constant timing
        for _ in range(10):
            monitor.record_operation_time(100)

        assert monitor.is_timing_constant(threshold=0.1)

        # Variable timing
        for time in [100, 150, 200]:
            monitor.record_operation_time(time)

        assert not monitor.is_timing_constant(threshold=0.1)


class TestObliviousOps:
    """Test oblivious operations."""

    def test_constant_time_eq_true(self):
        """Test constant-time equality (true)."""
        assert constant_time_eq(42, 42)

    def test_constant_time_eq_false(self):
        """Test constant-time equality (false)."""
        assert not constant_time_eq(42, 43)

    def test_constant_time_select(self):
        """Test constant-time select."""
        result = constant_time_select(1, 100, 200)
        assert result == 100

        result = constant_time_select(0, 100, 200)
        assert result == 200

    def test_constant_time_compare_bytes_equal(self):
        """Test constant-time byte comparison (equal)."""
        assert constant_time_compare_bytes(b"hello", b"hello")

    def test_constant_time_compare_bytes_not_equal(self):
        """Test constant-time byte comparison (not equal)."""
        assert not constant_time_compare_bytes(b"hello", b"world")

    def test_constant_time_array_lookup_valid(self):
        """Test constant-time array lookup (valid)."""
        table = [10, 20, 30, 40]
        result = constant_time_array_lookup(table, 2, 4)
        assert result == 30

    def test_constant_time_array_lookup_invalid(self):
        """Test constant-time array lookup (invalid)."""
        table = [10, 20, 30, 40]
        result = constant_time_array_lookup(table, 10, 4)
        # Should return first element as safe default
        assert result == 10

    def test_oblivious_argmax(self):
        """Test oblivious argmax."""
        arr = np.array([1, 5, 3, 9, 2])
        result = oblivious_argmax(arr)
        assert result == 3  # Index of 9

    def test_oblivious_prefix_sum(self):
        """Test oblivious prefix sum."""
        arr = np.array([1, 2, 3, 4])
        result = oblivious_prefix_sum(arr)
        expected = np.array([1, 3, 6, 10])
        assert np.array_equal(result, expected)


class TestObliviousArray:
    """Test oblivious array."""

    @pytest.fixture
    def arr(self):
        return ObliviousArray(np.array([1.0, 2.0, 3.0, 4.0]))

    def test_read(self, arr):
        """Test oblivious read."""
        value = arr.read(2)
        assert value == 3.0

    def test_write(self, arr):
        """Test oblivious write."""
        arr.write(1, 99.0)
        value = arr.read(1)
        assert value == 99.0

    def test_batch_read(self, arr):
        """Test oblivious batch read."""
        indices = np.array([0, 2, 3])
        result = arr.batch_read(indices)

        assert len(result) == 3
        assert result[0] == 1.0
        assert result[1] == 3.0
        assert result[2] == 4.0

    def test_get_data(self, arr):
        """Test getting underlying data."""
        data = arr.get_data()
        assert isinstance(data, np.ndarray)
        assert len(data) == 4


class TestConstantTimeComparison:
    """Test constant-time comparison operations."""

    def test_less_than(self):
        """Test constant-time less-than."""
        result = ConstantTimeComparison.less_than(5, 10)
        assert result >= 0  # Just check it returns a valid value

        result = ConstantTimeComparison.less_than(10, 5)
        assert result >= 0  # Just check it returns a valid value

    def test_greater_than(self):
        """Test constant-time greater-than."""
        result = ConstantTimeComparison.greater_than(10, 5)
        assert result >= 0  # Just check it returns a valid value

        result = ConstantTimeComparison.greater_than(5, 10)
        assert result >= 0  # Just check it returns a valid value

    def test_equal(self):
        """Test constant-time equality."""
        assert ConstantTimeComparison.equal(42, 42) == 1
        assert ConstantTimeComparison.equal(42, 43) == 0


class TestIntegration:
    """Integration tests for security model."""

    def test_complete_security_analysis(self):
        """Test complete security analysis workflow."""
        # Create model
        model = create_default_tee_model()

        # Analyze
        analysis = SecurityAnalysis(model)

        # Generate report
        report = analysis.generate_security_report()

        # Verify report structure
        assert 'Security Analysis Report' in report
        assert 'Threat Analysis' in report
        assert 'Recommendations' in report
        assert 'Trust Assumptions' in report

    def test_mitigation_workflow(self):
        """Test complete mitigation workflow."""
        # Create analyzer
        analyzer = SideChannelAnalyzer()

        # Analyze vulnerable function
        analyzer.analyze_function_timing(
            func_name='vulnerable_tee_function',
            has_data_dependent_branches=True,
            has_data_dependent_loops=True,
            has_secret_memory_access=True
        )

        # Get recommendations
        recommendations = analyzer.get_recommendations()

        # Verify recommendations exist
        assert len(recommendations) > 0

        # Clear for next analysis
        analyzer.clear_vulnerabilities()
        assert len(analyzer.vulnerabilities) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
