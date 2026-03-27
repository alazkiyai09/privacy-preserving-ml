"""
Security Model and Threat Analysis for TEE
============================================

Defines threat models, security properties, and attack vectors
for Trusted Execution Environments.

Key Concepts:
- Threat actors (who might attack us)
- Attack vectors (how they might attack)
- Security properties (what we protect)
- Limitations (what TEE doesn't protect)
"""

from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
from enum import Enum


class ThreatActor(Enum):
    """
    Types of threat actors in TEE systems.

    From most to least powerful:
    """

    MALICIOUS_OS = "malicious_os"
    """
    Compromised operating system or hypervisor.
    - Can read/write all memory (except enclave)
    - Can inject code (except into enclave)
    - Can perform side-channel attacks
    - CANNOT: Read enclave memory, tamper with enclave execution
    """

    MALICIOUS_HARDWARE_VENDOR = "malicious_hardware_vendor"
    """
    Hardware manufacturer (Intel, ARM, etc.) acting maliciously.
    - Has backdoor access to everything
    - Can compromise attestation
    - CANNOT: None (this is the ultimate threat)
    """

    MALICIOUS_APPLICATION = "malicious_application"
    """
    Other applications running on same system.
    - Can use system calls
    - Can attempt side-channel attacks
    - CANNOT: Access enclave memory, inject code into enclave
    """

    MALICIOUS_CLIENT = "malicious_client"
    """
    Client sending malicious inputs.
    - Can send malformed data
    - Can attempt protocol-level attacks
    - CANNOT: Access enclave memory (if attestation works)
    """

    HONEST_BUT_CURIOUS = "honest_but_curious"
    """
    Server following protocol but curious about data.
    - Follows protocol correctly
    - Wants to learn sensitive information
    - CANNOT: Access enclave memory (if encryption works)
    """

    NETWORK_ATTACKER = "network_attacker"
    """
    Attacker with network access only.
    - Can intercept/modify network traffic
    - Can perform replay attacks
    - CANNOT: Access enclave memory, compromise attestation
    """


class AttackVector(Enum):
    """
    Methods by which threat actors might compromise TEE security.
    """

    # Direct Attacks
    MEMORY_SNOOPING = "memory_snooping"
    """
    Reading enclave memory from outside.
    Protected by: Memory encryption (EPC)
    """

    CODE_TAMPERING = "code_tampering"
    """
    Modifying enclave code.
    Protected by: Measurement verification (attestation)
    """

    # Side-Channel Attacks
    CACHE_TIMING = "cache_timing"
    """
    Cache-based side channels (Prime+Probe, Flush+Reload).
    Protected by: Constant-time operations, cache randomization
    """

    POWER_ANALYSIS = "power_analysis"
    """
    Power consumption side channels.
    Protected by: Constant-time operations, noise injection
    """

    TIMING_ATTACKS = "timing_attacks"
    """
    Execution time side channels.
    Protected by: Constant-time operations
    """

    SPECTRE_MELTDOWN = "spectre_meltdown"
    """
    Speculative execution attacks.
    Protected by: Software mitigations, CPU microcode updates
    """

    # Software Attacks
    IAGO_ATTACKS = "iago_attacks"
    """
    Malicious inputs that cause protocol violations.
    Protected by: Input validation, protocol design
    """

    REPLAY_ATTACKS = "replay_attacks"
    """
    Reusing old valid messages.
    Protected by: Nonces, timestamps
    """

    DENIAL_OF_SERVICE = "denial_of_service"
    """
    Exhausting enclave resources.
    Protected by: Resource limits, rate limiting
    """

    # Attestation Attacks
    FAKE_ATTESTATION = "fake_attestation"
    """
    Forging attestation reports.
    Protected by: Hardware-rooted keys, IAS verification
    """

    MALICIOUS_ENCLAVE = "malicious_enclave"
    """
    Enclave running malicious code.
    Protected by: Code review, measurement verification
    """


class SecurityProperty(Enum):
    """
    Security properties provided by TEE (in theory).
    """

    CONFIDENTIALITY = "confidentiality"
    """
    Data cannot be read from outside enclave.
    Provided by: Memory encryption (EPC)
    """

    INTEGRITY = "integrity"
    """
    Code and data cannot be modified from outside enclave.
    Provided by: Memory encryption, measurement
    """

    ISOLATION = "isolation"
    """
    Enclave execution is isolated from main OS.
    Provided by: Hardware-enforced boundaries
    """

    ATTESTATION = "attestation"
    """
    Can prove to remote party what code is running.
    Provided by: Remote attestation protocol
    """

    SEALED_STORAGE = "sealed_storage"
    """
    Data can be encrypted to specific enclave.
    Provided by: Seal keys derived from measurement
    """


@dataclass
class SecurityCapability:
    """
    What a specific threat actor can and cannot do.
    """

    actor: ThreatActor
    can_read_enclave_memory: bool
    can_modify_enclave_code: bool
    can_perform_side_channels: bool
    can_compromise_attestation: bool
    can_intercept_network: bool
    can_send_malicious_inputs: bool

    def get_attack_vectors(self) -> List[AttackVector]:
        """Get list of viable attack vectors for this actor."""
        vectors = []

        if self.can_perform_side_channels:
            vectors.extend([
                AttackVector.CACHE_TIMING,
                AttackVector.POWER_ANALYSIS,
                AttackVector.TIMING_ATTACKS,
            ])

        if self.can_compromise_attestation:
            vectors.append(AttackVector.FAKE_ATTESTATION)

        if self.can_intercept_network:
            vectors.extend([
                AttackVector.REPLAY_ATTACKS,
                AttackVector.IAGO_ATTACKS,
            ])

        if self.can_send_malicious_inputs:
            vectors.extend([
                AttackVector.IAGO_ATTACKS,
                AttackVector.DENIAL_OF_SERVICE,
            ])

        return vectors


@dataclass
class Protection:
    """
    How TEE protects against specific attacks.
    """

    attack: AttackVector
    protection_mechanism: str
    effectiveness: str  # 'complete', 'partial', 'minimal'
    description: str


@dataclass
class ThreatModel:
    """
    Complete threat model for TEE system.

    Defines:
    - Who might attack us (actors)
    - How they might attack (vectors)
    - What protects us (protections)
    - What we're still vulnerable to (limitations)
    """

    name: str
    description: str

    # Threat actors
    actors: List[ThreatActor] = field(default_factory=list)

    # Security properties
    properties: List[SecurityProperty] = field(default_factory=list)

    # Protections
    protections: List[Protection] = field(default_factory=list)

    # Limitations
    limitations: List[str] = field(default_factory=list)

    # Trust assumptions
    trust_assumptions: List[str] = field(default_factory=list)

    def is_protected_against(self, attack: AttackVector) -> bool:
        """Check if we have protection against specific attack."""
        return any(p.attack == attack for p in self.protections)

    def get_protection(self, attack: AttackVector) -> Optional[Protection]:
        """Get protection mechanism for specific attack."""
        for p in self.protections:
            if p.attack == attack:
                return p
        return None

    def get_actor_capabilities(self, actor: ThreatActor) -> SecurityCapability:
        """Get capabilities for specific threat actor."""
        capabilities = {
            ThreatActor.MALICIOUS_OS: SecurityCapability(
                actor=ThreatActor.MALICIOUS_OS,
                can_read_enclave_memory=False,  # Protected by EPC
                can_modify_enclave_code=False,  # Protected by EPC
                can_perform_side_channels=True,  # Possible
                can_compromise_attestation=False,  # Hardware-rooted
                can_intercept_network=True,  # Controls network stack
                can_send_malicious_inputs=False,  # Not a client
            ),
            ThreatActor.MALICIOUS_HARDWARE_VENDOR: SecurityCapability(
                actor=ThreatActor.MALICIOUS_HARDWARE_VENDOR,
                can_read_enclave_memory=True,  # Backdoor access
                can_modify_enclave_code=True,  # Backdoor access
                can_perform_side_channels=True,
                can_compromise_attestation=True,  # Controls attestation
                can_intercept_network=False,  # Not on the machine
                can_send_malicious_inputs=False,
            ),
            ThreatActor.MALICIOUS_APPLICATION: SecurityCapability(
                actor=ThreatActor.MALICIOUS_APPLICATION,
                can_read_enclave_memory=False,  # Protected by EPC
                can_modify_enclave_code=False,  # Protected by EPC
                can_perform_side_channels=True,  # Possible (cache attacks)
                can_compromise_attestation=False,
                can_intercept_network=False,
                can_send_malicious_inputs=False,
            ),
            ThreatActor.MALICIOUS_CLIENT: SecurityCapability(
                actor=ThreatActor.MALICIOUS_CLIENT,
                can_read_enclave_memory=False,
                can_modify_enclave_code=False,
                can_perform_side_channels=False,  # Remote
                can_compromise_attestation=False,
                can_intercept_network=True,  # Can intercept network traffic
                can_send_malicious_inputs=True,
            ),
            ThreatActor.HONEST_BUT_CURIOUS: SecurityCapability(
                actor=ThreatActor.HONEST_BUT_CURIOUS,
                can_read_enclave_memory=False,  # Follows protocol
                can_modify_enclave_code=False,
                can_perform_side_channels=False,
                can_compromise_attestation=False,
                can_intercept_network=False,
                can_send_malicious_inputs=False,
            ),
            ThreatActor.NETWORK_ATTACKER: SecurityCapability(
                actor=ThreatActor.NETWORK_ATTACKER,
                can_read_enclave_memory=False,
                can_modify_enclave_code=False,
                can_perform_side_channels=False,
                can_compromise_attestation=False,
                can_intercept_network=True,
                can_send_malicious_inputs=True,
            ),
        }

        return capabilities.get(actor, SecurityCapability(
            actor=actor,
            can_read_enclave_memory=False,
            can_modify_enclave_code=False,
            can_perform_side_channels=False,
            can_compromise_attestation=False,
            can_intercept_network=False,
            can_send_malicious_inputs=False,
        ))

    def assess_risk(self, actor: ThreatActor, attack: AttackVector) -> str:
        """
        Assess risk level for specific actor/attack combination.

        Returns:
            'critical', 'high', 'medium', 'low', 'mitigated'
        """
        capabilities = self.get_actor_capabilities(actor)

        # Check if actor can perform this attack
        viable_attack = attack in capabilities.get_attack_vectors()

        if not viable_attack:
            return 'mitigated'

        # Check if we have protection
        protection = self.get_protection(attack)

        if protection is None:
            # No protection
            if actor in [ThreatActor.MALICIOUS_HARDWARE_VENDOR]:
                return 'critical'
            elif actor in [ThreatActor.MALICIOUS_OS]:
                return 'high'
            else:
                return 'medium'
        else:
            # Has protection
            if protection.effectiveness == 'complete':
                return 'mitigated'
            elif protection.effectiveness == 'partial':
                if actor in [ThreatActor.MALICIOUS_HARDWARE_VENDOR]:
                    return 'high'
                else:
                    return 'medium'
            else:  # minimal
                return 'high'

    def get_security_recommendations(self) -> List[str]:
        """Get security recommendations based on threat model."""
        recommendations = []

        # Check for side-channel vulnerabilities
        side_channel_attacks = [
            AttackVector.CACHE_TIMING,
            AttackVector.POWER_ANALYSIS,
            AttackVector.TIMING_ATTACKS,
        ]

        vulnerable = any(
            self.get_protection(attack) is None or
            self.get_protection(attack).effectiveness != 'complete'
            for attack in side_channel_attacks
        )

        if vulnerable:
            recommendations.append(
                "Implement constant-time operations for security-critical code"
            )
            recommendations.append(
                "Use cache randomization and oblivious RAM patterns"
            )

        # Check attestation
        if not self.is_protected_against(AttackVector.FAKE_ATTESTATION):
            recommendations.append(
                "Verify attestation reports with Intel Attestation Service (IAS)"
            )

        # Check network security
        if not self.is_protected_against(AttackVector.REPLAY_ATTACKS):
            recommendations.append(
                "Use nonces and timestamps to prevent replay attacks"
            )

        # Check input validation
        if not self.is_protected_against(AttackVector.IAGO_ATTACKS):
            recommendations.append(
                "Validate all inputs and enforce protocol invariants"
            )

        return recommendations


def create_default_tee_model() -> ThreatModel:
    """
    Create default TEE threat model.

    This model represents the standard security assumptions
    for Intel SGX / ARM TrustZone.

    Returns:
        ThreatModel with standard TEE security assumptions
    """
    model = ThreatModel(
        name="Standard TEE Model",
        description="Default threat model for Intel SGX / ARM TrustZone",
    )

    # Security properties
    model.properties = [
        SecurityProperty.CONFIDENTIALITY,
        SecurityProperty.INTEGRITY,
        SecurityProperty.ISOLATION,
        SecurityProperty.ATTESTATION,
        SecurityProperty.SEALED_STORAGE,
    ]

    # Threat actors
    model.actors = [
        ThreatActor.HONEST_BUT_CURIOUS,
        ThreatActor.MALICIOUS_CLIENT,
        ThreatActor.NETWORK_ATTACKER,
        ThreatActor.MALICIOUS_OS,
        ThreatActor.MALICIOUS_APPLICATION,
        # ThreatActor.MALICIOUS_HARDWARE_VENDOR,  # Usually excluded
    ]

    # Protections
    model.protections = [
        Protection(
            attack=AttackVector.MEMORY_SNOOPING,
            protection_mechanism="Enclave Page Cache (EPC) encryption",
            effectiveness="complete",
            description="CPU encrypts enclave memory on every memory access",
        ),
        Protection(
            attack=AttackVector.CODE_TAMPERING,
            protection_mechanism="Measurement verification (attestation)",
            effectiveness="complete",
            description="Any code change changes measurement, detected by attestation",
        ),
        Protection(
            attack=AttackVector.REPLAY_ATTACKS,
            protection_mechanism="Nonces and timestamps in attestation",
            effectiveness="complete",
            description="Each attestation report includes unique nonce",
        ),
        Protection(
            attack=AttackVector.CACHE_TIMING,
            protection_mechanism="Constant-time operations (partial)",
            effectiveness="partial",
            description="Some protection through careful coding, but not complete",
        ),
        Protection(
            attack=AttackVector.SPECTRE_MELTDOWN,
            protection_mechanism="Software mitigations + microcode updates",
            effectiveness="partial",
            description="Known mitigations applied, but new variants may exist",
        ),
    ]

    # Limitations (what TEE doesn't protect against)
    model.limitations = [
        "Side-channel attacks (cache timing, power analysis)",
        "Speculative execution attacks (Spectre, Meltdown)",
        "Hardware bugs or backdoors",
        "Denial of service (resource exhaustion)",
        "Iago attacks (malicious protocol inputs)",
        "Physical attacks on hardware",
    ]

    # Trust assumptions
    model.trust_assumptions = [
        "Hardware manufacturer is honest (Intel, ARM, etc.)",
        "Attestation service is honest (Intel Attestation Service)",
        "CPU microcode is correct",
        "Hardware implementation matches specification",
    ]

    return model


def create_ht2ml_threat_model() -> ThreatModel:
    """
    Create threat model for HT2ML hybrid system.

    HT2ML combines HE and TEE, providing layered security:
    - HE: Cryptographic privacy for input
    - TEE: Hardware-enforced privacy for computation

    Returns:
        ThreatModel for HT2ML system
    """
    model = ThreatModel(
        name="HT2ML Hybrid Model",
        description="Threat model for hybrid HE/TEE system",
    )

    # Security properties (both HE and TEE)
    model.properties = [
        SecurityProperty.CONFIDENTIALITY,
        SecurityProperty.INTEGRITY,
        SecurityProperty.ISOLATION,
        SecurityProperty.ATTESTATION,
        SecurityProperty.SEALED_STORAGE,
    ]

    # Threat actors
    model.actors = [
        ThreatActor.HONEST_BUT_CURIOUS,
        ThreatActor.MALICIOUS_CLIENT,
        ThreatActor.MALICIOUS_OS,
    ]

    # Protections (HE + TEE)
    model.protections = [
        Protection(
            attack=AttackVector.MEMORY_SNOOPING,
            protection_mechanism="HE encryption + EPC encryption",
            effectiveness="complete",
            description="Input data encrypted with HE, computation in TEE",
        ),
        Protection(
            attack=AttackVector.CODE_TAMPERING,
            protection_mechanism="HE correctness + TEE measurement",
            effectiveness="complete",
            description="HE processing verified, TEE code attested",
        ),
        Protection(
            attack=AttackVector.REPLAY_ATTACKS,
            protection_mechanism="HE nonces + TEE attestation",
            effectiveness="complete",
            description="Dual protection against replay",
        ),
    ]

    # Limitations
    model.limitations = [
        "TEE side-channels still possible for TEE-processed data",
        "HE operations limited by noise budget",
        "Trust required in hardware for TEE portion",
        "Trust required in HE library correctness",
    ]

    # Trust assumptions
    model.trust_assumptions = [
        "HE cryptography is mathematically sound",
        "TEE hardware is honest (for TEE portion)",
        "HE→TEE handoff protocol is correctly implemented",
        "Attestation reports are verified",
    ]

    return model


class SecurityAnalysis:
    """
    Analyze security of TEE system.

    Provides methods for:
    - Risk assessment
    - Vulnerability analysis
    - Mitigation recommendations
    """

    def __init__(self, model: ThreatModel):
        """
        Initialize security analysis.

        Args:
            model: Threat model to analyze
        """
        self.model = model

    def analyze_threats(self) -> Dict[str, List[str]]:
        """
        Analyze threats and generate report.

        Returns:
            Dictionary with threat analysis
        """
        threats = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': [],
            'mitigated': [],
        }

        for actor in self.model.actors:
            for attack in AttackVector:
                risk = self.model.assess_risk(actor, attack)
                threats[risk].append(
                    f"{actor.value}: {attack.value}"
                )

        return threats

    def get_vulnerabilities(self) -> List[str]:
        """
        Get list of unmitigated vulnerabilities.

        Returns:
            List of vulnerability descriptions
        """
        vulnerabilities = []

        for actor in self.model.actors:
            for attack in AttackVector:
                risk = self.model.assess_risk(actor, attack)
                if risk in ['critical', 'high', 'medium']:
                    vulnerabilities.append(
                        f"[{risk.upper()}] {actor.value} could perform {attack.value}"
                    )

        return vulnerabilities

    def recommend_mitigations(self) -> List[str]:
        """
        Get mitigation recommendations.

        Returns:
            List of recommendations
        """
        return self.model.get_security_recommendations()

    def verify_isolation(self) -> bool:
        """
        Verify that enclave provides isolation.

        In simulation, this always returns True.
        In production, would run actual tests.

        Returns:
            True (simulated isolation)
        """
        # In real deployment, would:
        # 1. Try to read enclave memory from outside (should fail)
        # 2. Try to modify enclave code (should fail)
        # 3. Verify EPC boundaries are enforced

        return True  # Simulated

    def verify_attestation(self) -> bool:
        """
        Verify that attestation works correctly.

        In simulation, this checks the mock attestation.

        Returns:
            True (simulated attestation)
        """
        # In real deployment, would:
        # 1. Generate attestation report
        # 2. Verify signature
        # 3. Check measurement
        # 4. Verify freshness (nonce, timestamp)

        return True  # Simulated

    def generate_security_report(self) -> str:
        """
        Generate comprehensive security report.

        Returns:
            Security report as string
        """
        report = []
        report.append("=" * 70)
        report.append("TEE Security Analysis Report")
        report.append("=" * 70)
        report.append("")

        # Threat analysis
        threats = self.analyze_threats()
        report.append("## Threat Analysis")
        report.append("")

        for level, threat_list in threats.items():
            if threat_list:
                report.append(f"{level.upper()} ({len(threat_list)}):")
                for threat in threat_list:
                    report.append(f"  - {threat}")
                report.append("")

        # Vulnerabilities
        vulnerabilities = self.get_vulnerabilities()
        if vulnerabilities:
            report.append("## Unmitigated Vulnerabilities")
            report.append("")
            for vuln in vulnerabilities:
                report.append(f"  {vuln}")
            report.append("")

        # Recommendations
        recommendations = self.recommend_mitigations()
        if recommendations:
            report.append("## Security Recommendations")
            report.append("")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")

        # Trust assumptions
        report.append("## Trust Assumptions")
        report.append("")
        for assumption in self.model.trust_assumptions:
            report.append(f"- {assumption}")
        report.append("")

        report.append("=" * 70)

        return "\n".join(report)
