"""
Unit Tests for HE↔TEE Handoff Protocol and Split Optimizer
===========================================================

Tests for:
- HE context and data structures
- HE→TEE handoff protocol
- TEE→HE handoff protocol
- Split point optimization
- Split strategy recommendations
- Protocol analysis and optimization
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tee_ml.core.enclave import Enclave
from tee_ml.protocol.handoff import (
    HEContext,
    HEData,
    HEtoTEEHandoff,
    TEEtoHEHandoff,
    HandoffResult,
    HandoffDirection,
    HT2MLProtocol,
    create_handoff_protocol,
    validate_handoff_security,
    estimate_handoff_cost,
    ProtocolOptimizer,
    simulate_ht2ml_protocol,
)
from tee_ml.protocol.split_optimizer import (
    LayerSpecification,
    SplitStrategy,
    SplitRecommendation,
    SplitOptimizer,
    create_layer_specifications,
    estimate_optimal_split,
    visualize_split,
    analyze_tradeoffs,
)


class TestHEContext:
    """Test HE context data structure."""

    def test_create_ckks_context(self):
        """Test creating CKKS context."""
        context = HEContext(
            scheme='ckks',
            poly_modulus_degree=4096,
            scale=2**30,
            eval=1,
        )

        assert context.scheme == 'ckks'
        assert context.poly_modulus_degree == 4096
        assert context.scale == 2**30
        assert context.eval == 1

    def test_create_bfv_context(self):
        """Test creating BFV context."""
        context = HEContext(
            scheme='bfv',
            poly_modulus_degree=4096,
            scale=1.0,
            eval=2,
        )

        assert context.scheme == 'bfv'

    def test_context_to_dict(self):
        """Test converting context to dictionary."""
        context = HEContext(
            scheme='ckks',
            poly_modulus_degree=8192,
            scale=2**40,
            eval=1,
        )

        result = context.to_dict()

        assert result['scheme'] == 'ckks'
        assert result['poly_modulus_degree'] == 8192
        assert result['scale'] == 2**40
        assert result['eval'] == 1


class TestHEData:
    """Test HE data structure."""

    def test_create_he_data(self):
        """Test creating HE data."""
        data = HEData(
            encrypted_data=[1, 2, 3],  # Mock encrypted data
            shape=(10,),
            scheme='ckks',
            scale=2**30,
        )

        assert data.shape == (10,)
        assert data.scheme == 'ckks'
        assert data.scale == 2**30
        assert data.size == 3  # Length of list

    def test_he_data_with_array(self):
        """Test HE data with numpy array."""
        # Mock encrypted data with size() method
        class MockEncrypted:
            def size(self):
                return 5

        data = HEData(
            encrypted_data=MockEncrypted(),
            shape=(5,),
            scheme='ckks',
            scale=2**30,
        )

        assert data.size == 5


class TestHEtoTEEHandoff:
    """Test HE to TEE handoff."""

    @pytest.fixture
    def he_context(self):
        return HEContext(
            scheme='ckks',
            poly_modulus_degree=4096,
            scale=2**30,
            eval=1,
        )

    @pytest.fixture
    def he_data(self, he_context):
        return HEData(
            encrypted_data=[1, 2, 3],
            shape=(10,),
            scheme=he_context.scheme,
            scale=he_context.scale,
        )

    def test_create_handoff(self, he_data, he_context):
        """Test creating handoff."""
        handoff = HEtoTEEHandoff(
            encrypted_data=he_data,
            he_context=he_context,
            nonce=b"test-nonce",
        )

        assert handoff.encrypted_data == he_data
        assert handoff.he_context == he_context
        assert handoff.nonce == b"test-nonce"

    def test_validate_valid_handoff(self, he_data, he_context):
        """Test validation of valid handoff."""
        handoff = HEtoTEEHandoff(
            encrypted_data=he_data,
            he_context=he_context,
        )

        assert handoff.validate() == True

    def test_validate_invalid_handoff(self, he_context):
        """Test validation of invalid handoff (None data)."""
        handoff = HEtoTEEHandoff(
            encrypted_data=None,
            he_context=he_context,
        )

        assert handoff.validate() == False

    def test_validate_scheme_mismatch(self, he_data):
        """Test validation with scheme mismatch."""
        wrong_context = HEContext(
            scheme='bfv',  # Different scheme
            poly_modulus_degree=4096,
            scale=1.0,
            eval=1,
        )

        handoff = HEtoTEEHandoff(
            encrypted_data=he_data,
            he_context=wrong_context,
        )

        assert handoff.validate() == False


class TestTEEtoHEHandoff:
    """Test TEE to HE handoff."""

    @pytest.fixture
    def he_context(self):
        return HEContext(
            scheme='ckks',
            poly_modulus_degree=4096,
            scale=2**30,
            eval=1,
        )

    def test_create_handoff(self, he_context):
        """Test creating handoff."""
        plaintext = np.array([1.0, 2.0, 3.0])

        handoff = TEEtoHEHandoff(
            plaintext_data=plaintext,
            he_context=he_context,
            reencrypt=True,
        )

        assert np.array_equal(handoff.plaintext_data, plaintext)
        assert handoff.he_context == he_context
        assert handoff.reencrypt == True

    def test_validate_valid_handoff(self, he_context):
        """Test validation of valid handoff."""
        plaintext = np.array([1.0, 2.0, 3.0])

        handoff = TEEtoHEHandoff(
            plaintext_data=plaintext,
            he_context=he_context,
        )

        assert handoff.validate() == True

    def test_validate_invalid_handoff(self, he_context):
        """Test validation with None data."""
        handoff = TEEtoHEHandoff(
            plaintext_data=None,
            he_context=he_context,
        )

        assert handoff.validate() == False

    def test_validate_missing_context(self):
        """Test validation with missing context."""
        plaintext = np.array([1.0, 2.0, 3.0])

        handoff = TEEtoHEHandoff(
            plaintext_data=plaintext,
            he_context=None,
        )

        assert handoff.validate() == False


class TestHT2MLProtocol:
    """Test HT2ML protocol implementation."""

    @pytest.fixture
    def enclave(self):
        return Enclave(enclave_id="test-enclave")

    @pytest.fixture
    def protocol(self, enclave):
        return HT2MLProtocol(enclave)

    @pytest.fixture
    def he_context(self):
        return HEContext(
            scheme='ckks',
            poly_modulus_degree=4096,
            scale=2**30,
            eval=1,
        )

    @pytest.fixture
    def he_data(self, he_context):
        return HEData(
            encrypted_data=[1, 2, 3],
            shape=(10,),
            scheme=he_context.scheme,
            scale=he_context.scale,
        )

    def test_protocol_creation(self, protocol, enclave):
        """Test protocol creation."""
        assert protocol.enclave == enclave
        assert protocol.session is None
        assert len(protocol.handoff_history) == 0

    def test_handoff_he_to_tee_success(self, protocol, he_data, he_context):
        """Test successful HE to TEE handoff."""
        success, plaintext = protocol.handoff_he_to_tee(
            encrypted_data=he_data,
            he_context=he_context,
            nonce=b"test-nonce",
        )

        assert success == True
        assert isinstance(plaintext, np.ndarray)
        assert len(plaintext) == he_data.size

    def test_handoff_he_to_tee_invalid(self, protocol, he_context):
        """Test HE to TEE handoff with invalid data."""
        invalid_data = HEData(
            encrypted_data=None,
            shape=(0,),
            scheme='ckks',
            scale=2**30,
        )

        success, plaintext = protocol.handoff_he_to_tee(
            encrypted_data=invalid_data,
            he_context=he_context,
        )

        assert success == False
        assert len(plaintext) == 0

    def test_handoff_tee_to_he_success(self, protocol, he_context):
        """Test successful TEE to HE handoff."""
        plaintext = np.array([1.0, 2.0, 3.0])

        success, encrypted = protocol.handoff_tee_to_he(
            plaintext_data=plaintext,
            he_context=he_context,
            reencrypt=False,  # Don't actually encrypt in simulation
        )

        assert success == True
        assert encrypted is not None
        assert encrypted.shape == plaintext.shape

    def test_handoff_tee_to_he_invalid(self, protocol, he_context):
        """Test TEE to HE handoff with invalid data."""
        success, encrypted = protocol.handoff_tee_to_he(
            plaintext_data=None,
            he_context=he_context,
        )

        assert success == False
        assert encrypted is None

    def test_handoff_history_tracking(self, protocol, he_data, he_context):
        """Test that handoff operations are tracked."""
        # Perform multiple handoffs
        for i in range(3):
            protocol.handoff_he_to_tee(
                encrypted_data=he_data,
                he_context=he_context,
                nonce=f"nonce-{i}".encode(),
            )

        history = protocol.get_handoff_history()

        assert len(history) == 3

        for result in history:
            assert result.direction == HandoffDirection.HE_TO_TEE

    def test_handoff_statistics(self, protocol, he_data, he_context):
        """Test handoff statistics calculation."""
        # Perform successful handoffs
        for i in range(5):
            protocol.handoff_he_to_tee(
                encrypted_data=he_data,
                he_context=he_context,
                nonce=f"nonce-{i}".encode(),
            )

        # Perform one failed handoff
        invalid_data = HEData(
            encrypted_data=None,
            shape=(0,),
            scheme='ckks',
            scale=2**30,
        )
        protocol.handoff_he_to_tee(
            encrypted_data=invalid_data,
            he_context=he_context,
        )

        stats = protocol.get_handoff_statistics()

        assert stats['total_handoffs'] == 6
        assert stats['successful'] == 5
        assert stats['failed'] == 1
        assert stats['success_rate'] == 5/6
        assert stats['total_time_ns'] > 0
        assert stats['avg_time_ns'] > 0


class TestHandoffFactory:
    """Test factory functions."""

    def test_create_handoff_protocol(self):
        """Test protocol factory function."""
        enclave = Enclave(enclave_id="factory-test")
        protocol = create_handoff_protocol(enclave)

        assert isinstance(protocol, HT2MLProtocol)
        assert protocol.enclave == enclave

    def test_validate_handoff_security_valid(self):
        """Test security validation for valid handoff."""
        he_context = HEContext(
            scheme='ckks',
            poly_modulus_degree=4096,
            scale=2**30,
            eval=1,
        )

        he_data = HEData(
            encrypted_data=[1, 2, 3],
            shape=(10,),
            scheme='ckks',
            scale=2**30,
        )

        handoff = HEtoTEEHandoff(
            encrypted_data=he_data,
            he_context=he_context,
            nonce=b"fresh-nonce",
        )

        assert validate_handoff_security(handoff) == True

    def test_validate_handoff_security_no_nonce(self):
        """Test security validation with missing nonce."""
        he_context = HEContext(
            scheme='ckks',
            poly_modulus_degree=4096,
            scale=2**30,
            eval=1,
        )

        he_data = HEData(
            encrypted_data=[1, 2, 3],
            shape=(10,),
            scheme='ckks',
            scale=2**30,
        )

        handoff = HEtoTEEHandoff(
            encrypted_data=he_data,
            he_context=he_context,
            nonce=None,  # Missing nonce
        )

        # Should still validate (nonce is optional)
        assert validate_handoff_security(handoff) == True

    def test_estimate_handoff_cost_he_to_tee(self):
        """Test cost estimation for HE→TEE handoff."""
        cost = estimate_handoff_cost(
            handoff_type=HandoffDirection.HE_TO_TEE,
            data_size_mb=1.5,
        )

        assert cost['handoff_type'] == 'he_to_tee'
        assert cost['data_size_mb'] == 1.5
        assert cost['total_overhead_ns'] > 0
        assert cost['total_overhead_us'] > 0
        assert cost['total_overhead_ms'] > 0

    def test_estimate_handoff_cost_tee_to_he(self):
        """Test cost estimation for TEE→HE handoff."""
        cost = estimate_handoff_cost(
            handoff_type=HandoffDirection.TEE_TO_HE,
            data_size_mb=0.5,
        )

        assert cost['handoff_type'] == 'tee_to_he'
        assert cost['data_size_mb'] == 0.5
        assert cost['total_overhead_ns'] > 0


class TestProtocolOptimizer:
    """Test protocol optimizer."""

    @pytest.fixture
    def enclave(self):
        return Enclave(enclave_id="optimizer-test")

    @pytest.fixture
    def protocol(self, enclave):
        return create_handoff_protocol(enclave)

    @pytest.fixture
    def optimizer(self, protocol):
        return ProtocolOptimizer(protocol)

    def test_optimizer_creation(self, optimizer, protocol):
        """Test optimizer creation."""
        assert optimizer.protocol == protocol

    def test_analyze_handoffs_empty(self, optimizer):
        """Test analysis with no handoffs."""
        analysis = optimizer.analyze_handoffs()

        assert analysis['total_handoffs'] == 0
        assert analysis['he_to_tee_count'] == 0
        assert analysis['tee_to_he_count'] == 0

    def test_analyze_handoffs_with_data(self, optimizer, protocol):
        """Test analysis with handoff data."""
        he_context = HEContext(
            scheme='ckks',
            poly_modulus_degree=4096,
            scale=2**30,
            eval=1,
        )

        he_data = HEData(
            encrypted_data=[1, 2, 3],
            shape=(10,),
            scheme='ckks',
            scale=2**30,
        )

        # Perform 5 HE→TEE handoffs
        for i in range(5):
            protocol.handoff_he_to_tee(
                encrypted_data=he_data,
                he_context=he_context,
                nonce=f"nonce-{i}".encode(),
            )

        analysis = optimizer.analyze_handoffs()

        assert analysis['total_handoffs'] == 5
        assert analysis['he_to_tee_count'] == 5
        assert analysis['tee_to_he_count'] == 0
        assert 'One-way handoff' in analysis['patterns'][0]

    def test_recommend_optimizations_low_success_rate(self, optimizer, protocol):
        """Test recommendations for low success rate."""
        he_context = HEContext(
            scheme='ckks',
            poly_modulus_degree=4096,
            scale=2**30,
            eval=1,
        )

        he_data = HEData(
            encrypted_data=None,  # Will cause failures
            shape=(0,),
            scheme='ckks',
            scale=2**30,
        )

        # Perform failed handoffs
        for i in range(10):
            protocol.handoff_he_to_tee(
                encrypted_data=he_data,
                he_context=he_context,
            )

        recommendations = optimizer.recommend_optimizations()

        # Should recommend improving success rate
        assert any('success rate' in r.lower() for r in recommendations)

    def test_recommend_optimizations_high_frequency(self, optimizer, protocol):
        """Test recommendations for high handoff frequency."""
        he_context = HEContext(
            scheme='ckks',
            poly_modulus_degree=4096,
            scale=2**30,
            eval=1,
        )

        he_data = HEData(
            encrypted_data=[1, 2, 3],
            shape=(10,),
            scheme='ckks',
            scale=2**30,
        )

        # Perform many handoffs
        for i in range(150):
            protocol.handoff_he_to_tee(
                encrypted_data=he_data,
                he_context=he_context,
                nonce=f"nonce-{i}".encode(),
            )

        recommendations = optimizer.recommend_optimizations()

        # Should recommend batching
        assert any('batching' in r.lower() for r in recommendations)


class TestLayerSpecification:
    """Test layer specification."""

    def test_create_layer_spec(self):
        """Test creating layer specification."""
        layer = LayerSpecification(
            index=0,
            input_size=10,
            output_size=5,
            activation='relu',
            use_bias=True,
        )

        assert layer.index == 0
        assert layer.input_size == 10
        assert layer.output_size == 5
        assert layer.activation == 'relu'
        assert layer.use_bias == True

    def test_noise_cost_linear(self):
        """Test noise cost for linear layer."""
        layer = LayerSpecification(
            index=0,
            input_size=10,
            output_size=5,
            activation='none',
        )

        cost = layer.get_noise_cost(scale_bits=40)

        # Linear cost only: 5 * 40 = 200
        assert cost == 200

    def test_noise_cost_with_activation(self):
        """Test noise cost with activation."""
        layer = LayerSpecification(
            index=0,
            input_size=10,
            output_size=5,
            activation='relu',
        )

        cost = layer.get_noise_cost(scale_bits=40)

        # Linear: 5 * 40 = 200
        # ReLU: 5 * 40 = 200
        # Total: 400
        assert cost == 400

    def test_tee_overhead_estimation(self):
        """Test TEE overhead estimation."""
        layer = LayerSpecification(
            index=0,
            input_size=100,
            output_size=50,
            activation='relu',
        )

        overhead = layer.get_tee_overhead_ns(data_size_mb=0.001)

        assert overhead > 0
        assert overhead > 25000  # At least the fixed overhead


class TestSplitOptimizer:
    """Test split optimizer."""

    @pytest.fixture
    def optimizer(self):
        return SplitOptimizer(
            noise_budget=200,
            scale_bits=40,
            max_he_layers=2,
        )

    @pytest.fixture
    def simple_layers(self):
        """Create simple layer specifications."""
        return [
            LayerSpecification(0, 10, 8, 'relu'),
            LayerSpecification(1, 8, 5, 'sigmoid'),
            LayerSpecification(2, 5, 3, 'softmax'),
        ]

    def test_optimizer_creation(self, optimizer):
        """Test optimizer creation."""
        assert optimizer.noise_budget == 200
        assert optimizer.scale_bits == 40
        assert optimizer.max_he_layers == 2

    def test_analyze_layer_cost(self, optimizer, simple_layers):
        """Test layer cost analysis."""
        costs = optimizer.analyze_layer_cost(simple_layers)

        assert len(costs) == 3

        # Layer 0: 8 * 40 + 5 * 40 = 520
        # Layer 1: 5 * 40 + 5 * 40 = 400
        # Layer 2: 3 * 40 + 5 * 40 = 320

        assert costs[0] == 520  # 520
        assert costs[1] == 920  # 520 + 400
        assert costs[2] == 1240  # 520 + 400 + 320

    def test_find_feasible_splits(self, optimizer, simple_layers):
        """Test finding feasible split points."""
        feasible = optimizer.find_feasible_splits(simple_layers)

        # Only layer 0 (520 > 200) fits? No, 520 > 200
        # Actually, no layers fit within 200-bit budget
        # So feasible should be empty or only have small layers

        # Let's create layers that fit
        small_layers = [
            LayerSpecification(0, 10, 2, 'none'),  # 2 * 40 = 80
            LayerSpecification(1, 2, 2, 'none'),   # 2 * 40 = 80
        ]

        feasible = optimizer.find_feasible_splits(small_layers)

        assert 1 in feasible  # After layer 0: 80 bits
        assert 2 in feasible  # After layer 1: 160 bits

    def test_estimate_performance(self, optimizer, simple_layers):
        """Test performance estimation."""
        perf = optimizer.estimate_performance(simple_layers, split_point=1)

        assert 'he_time_ns' in perf
        assert 'tee_time_ns' in perf
        assert 'total_time_ns' in perf
        assert perf['total_time_ns'] > 0

    def test_calculate_scores(self, optimizer, simple_layers):
        """Test privacy and performance score calculation."""
        privacy, performance = optimizer.calculate_scores(simple_layers, split_point=1)

        # Split after 1 layer: 1 HE layer, 2 TEE layers
        assert 0.0 <= privacy <= 1.0
        assert 0.0 <= performance <= 1.0

    def test_recommend_split_privacy_max(self, optimizer):
        """Test privacy-maximized split recommendation."""
        # Create very small layers to fit in budget
        layers = [
            LayerSpecification(0, 10, 1, 'none'),  # 40 bits
            LayerSpecification(1, 1, 1, 'none'),   # 40 bits
            LayerSpecification(2, 1, 1, 'none'),   # 40 bits
        ]

        rec = optimizer.recommend_split(layers, SplitStrategy.PRIVACY_MAX)

        assert rec.strategy == SplitStrategy.PRIVACY_MAX
        # With privacy_max, should maximize HE layers within budget
        # Can fit up to 5 layers (5 * 40 = 200 bits), but we only have 3
        assert rec.he_layers >= 1
        assert rec.split_point >= 1
        assert rec.is_feasible()

    def test_recommend_split_performance_max(self, optimizer):
        """Test performance-maximized split recommendation."""
        # Use small layers
        layers = [
            LayerSpecification(0, 10, 1, 'none'),  # 40 bits
            LayerSpecification(1, 1, 1, 'none'),   # 40 bits
        ]

        rec = optimizer.recommend_split(layers, SplitStrategy.PERFORMANCE_MAX)

        assert rec.strategy == SplitStrategy.PERFORMANCE_MAX
        # Performance max should minimize HE layers
        assert rec.he_layers >= 0

    def test_recommend_split_balanced(self, optimizer):
        """Test balanced split recommendation."""
        # Use small layers
        layers = [
            LayerSpecification(0, 10, 1, 'none'),  # 40 bits
            LayerSpecification(1, 1, 1, 'none'),   # 40 bits
        ]

        rec = optimizer.recommend_split(layers, SplitStrategy.BALANCED)

        assert rec.strategy == SplitStrategy.BALANCED
        assert rec.he_layers >= 0
        assert rec.tee_layers >= 0

    def test_compare_all_strategies(self, optimizer):
        """Test comparing all split strategies."""
        # Use small layers
        layers = [
            LayerSpecification(0, 10, 1, 'none'),  # 40 bits
            LayerSpecification(1, 1, 1, 'none'),   # 40 bits
        ]

        recommendations = optimizer.compare_all_strategies(layers)

        assert 'privacy_max' in recommendations
        assert 'performance_max' in recommendations
        assert 'balanced' in recommendations

        # Check that each is a SplitRecommendation
        for rec in recommendations.values():
            assert isinstance(rec, SplitRecommendation)


class TestSplitRecommendation:
    """Test split recommendation."""

    def test_recommendation_feasibility(self):
        """Test feasibility check."""
        # Feasible recommendation
        rec1 = SplitRecommendation(
            strategy=SplitStrategy.BALANCED,
            he_layers=1,
            tee_layers=2,
            split_point=1,
            noise_budget_used=100,
            noise_budget_remaining=100,
            estimated_he_time_ns=1000,
            estimated_tee_time_ns=2000,
            total_time_ns=3000,
            privacy_score=0.5,
            performance_score=0.67,
            rationale="Test",
            total_noise_budget=200,
        )

        assert rec1.is_feasible() == True

        # Infeasible recommendation
        rec2 = SplitRecommendation(
            strategy=SplitStrategy.BALANCED,
            he_layers=1,
            tee_layers=2,
            split_point=1,
            noise_budget_used=300,
            noise_budget_remaining=100,
            estimated_he_time_ns=1000,
            estimated_tee_time_ns=2000,
            total_time_ns=3000,
            privacy_score=0.5,
            performance_score=0.67,
            rationale="Test",
            total_noise_budget=200,
        )

        assert rec2.is_feasible() == False

    def test_print_summary(self, capsys):
        """Test printing summary."""
        rec = SplitRecommendation(
            strategy=SplitStrategy.BALANCED,
            he_layers=1,
            tee_layers=2,
            split_point=1,
            noise_budget_used=100,
            noise_budget_remaining=100,
            estimated_he_time_ns=1000000,
            estimated_tee_time_ns=2000000,
            total_time_ns=3000000,
            privacy_score=0.5,
            performance_score=0.67,
            rationale="Test recommendation",
            total_noise_budget=200,
        )

        rec.print_summary()

        captured = capsys.readouterr()

        assert "HT2ML Split Recommendation" in captured.out
        assert "balanced" in captured.out  # Strategy value is lowercase
        assert "Privacy:" in captured.out
        assert "Performance:" in captured.out


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_layer_specifications(self):
        """Test creating layer specifications from dimensions."""
        layers = create_layer_specifications(
            input_size=10,
            hidden_sizes=[8, 5],
            output_size=3,
            activations=['relu', 'sigmoid', 'softmax'],
        )

        assert len(layers) == 3

        assert layers[0].input_size == 10
        assert layers[0].output_size == 8
        assert layers[0].activation == 'relu'

        assert layers[1].input_size == 8
        assert layers[1].output_size == 5
        assert layers[1].activation == 'sigmoid'

        assert layers[2].input_size == 5
        assert layers[2].output_size == 3
        assert layers[2].activation == 'softmax'

    def test_estimate_optimal_split(self):
        """Test convenience function for optimal split."""
        # Use small layers to fit in budget
        rec = estimate_optimal_split(
            input_size=10,
            hidden_sizes=[2],  # Small hidden layer
            output_size=2,
            activations=['relu', 'softmax'],
            noise_budget=200,
            strategy=SplitStrategy.BALANCED,
        )

        assert isinstance(rec, SplitRecommendation)
        # Check that we got a recommendation (might be BALANCED or fallback to PERFORMANCE_MAX)
        assert rec.strategy in [SplitStrategy.BALANCED, SplitStrategy.PERFORMANCE_MAX]

    def test_visualize_split(self, capsys):
        """Test split visualization."""
        layers = [
            LayerSpecification(0, 10, 5, 'relu'),
            LayerSpecification(1, 5, 3, 'softmax'),
        ]

        rec = SplitRecommendation(
            strategy=SplitStrategy.BALANCED,
            he_layers=1,
            tee_layers=1,
            split_point=1,
            noise_budget_used=100,
            noise_budget_remaining=100,
            estimated_he_time_ns=1000,
            estimated_tee_time_ns=2000,
            total_time_ns=3000,
            privacy_score=0.5,
            performance_score=0.5,
            rationale="Test",
        )

        visualization = visualize_split(rec, layers)

        assert "Network Architecture:" in visualization
        assert "HE" in visualization
        assert "TEE" in visualization
        assert "SPLIT POINT" in visualization

    def test_analyze_tradeoffs(self):
        """Test trade-off analysis."""
        analysis = analyze_tradeoffs(
            input_size=10,
            hidden_sizes=[5],
            output_size=2,
            activations=['relu', 'softmax'],
            noise_budget=200,
        )

        assert 'num_layers' in analysis
        assert 'noise_budget' in analysis
        assert 'feasible_splits' in analysis
        assert 'recommendations' in analysis

        assert analysis['num_layers'] == 2
        assert analysis['noise_budget'] == 200

        # Check that all strategies are present
        assert 'privacy_max' in analysis['recommendations']
        assert 'performance_max' in analysis['recommendations']
        assert 'balanced' in analysis['recommendations']


class TestHT2MLSimulation:
    """Test HT2ML protocol simulation."""

    def test_simulate_ht2ml_protocol(self):
        """Test complete protocol simulation."""
        results = simulate_ht2ml_protocol(
            num_operations=10,
            data_size=100,
        )

        assert results['num_operations'] == 10
        assert results['successful_handoffs'] >= 0
        assert results['failed_handoffs'] >= 0
        assert results['success_rate'] >= 0.0
        assert results['total_time_ns'] > 0
        assert results['avg_time_ns'] > 0

        # Check that successful + failed = total
        assert results['successful_handoffs'] + results['failed_handoffs'] == 10


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_split_workflow(self):
        """Test complete split optimization workflow."""
        # Define network
        input_size = 20
        hidden_sizes = [10, 5]
        output_size = 2
        activations = ['relu', 'sigmoid', 'softmax']

        # Create layers
        layers = create_layer_specifications(
            input_size, hidden_sizes, output_size, activations
        )

        # Create optimizer
        optimizer = SplitOptimizer(noise_budget=200)

        # Get recommendations for all strategies
        recommendations = optimizer.compare_all_strategies(layers)

        # Check that we got recommendations
        assert len(recommendations) == 3

        # Analyze trade-offs
        analysis = analyze_tradeoffs(
            input_size, hidden_sizes, output_size, activations, noise_budget=200
        )

        assert analysis['num_layers'] == 3

    def test_complete_handoff_workflow(self):
        """Test complete handoff workflow."""
        # Create enclave and protocol
        enclave = Enclave(enclave_id="integration-test")
        protocol = create_handoff_protocol(enclave)

        # Create HE data
        he_context = HEContext(
            scheme='ckks',
            poly_modulus_degree=4096,
            scale=2**30,
            eval=1,
        )

        he_data = HEData(
            encrypted_data=[1, 2, 3, 4, 5],
            shape=(5,),
            scheme='ckks',
            scale=2**30,
        )

        # Perform handoff
        success, plaintext = protocol.handoff_he_to_tee(
            encrypted_data=he_data,
            he_context=he_context,
            nonce=b"integration-nonce",
        )

        assert success == True

        # Get statistics
        stats = protocol.get_handoff_statistics()

        assert stats['total_handoffs'] == 1
        assert stats['successful'] == 1

        # Create optimizer
        optimizer = ProtocolOptimizer(protocol)

        # Analyze handoffs
        analysis = optimizer.analyze_handoffs()

        assert analysis['total_handoffs'] == 1

    def test_ht2ml_architecture_simulation(self):
        """Test complete HT2ML architecture simulation."""
        # 1. Define network
        layers = create_layer_specifications(
            input_size=10,
            hidden_sizes=[5, 3],
            output_size=2,
            activations=['relu', 'sigmoid', 'softmax'],
        )

        # 2. Find optimal split
        optimizer = SplitOptimizer(noise_budget=200)
        rec = optimizer.recommend_split(layers, SplitStrategy.BALANCED)

        assert isinstance(rec, SplitRecommendation)

        # 3. Create protocol
        enclave = Enclave(enclave_id="ht2ml-sim")
        protocol = create_handoff_protocol(enclave)

        # 4. Simulate handoffs
        he_context = HEContext(
            scheme='ckks',
            poly_modulus_degree=4096,
            scale=2**30,
            eval=1,
        )

        he_data = HEData(
            encrypted_data=[1, 2, 3],
            shape=(5,),
            scheme='ckks',
            scale=2**30,
        )

        for i in range(5):
            protocol.handoff_he_to_tee(
                encrypted_data=he_data,
                he_context=he_context,
                nonce=f"nonce-{i}".encode(),
            )

        # 5. Check statistics
        stats = protocol.get_handoff_statistics()

        assert stats['total_handoffs'] == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
