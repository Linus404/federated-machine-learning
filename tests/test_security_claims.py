from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "SECURITY.md",
    ROOT / "THREAT_MODEL.md",
    ROOT / "COMPATIBILITY.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "docs" / "adr" / "0001-secure-aggregation.md",
    ROOT / "docs" / "SECURITY_PRIVACY_REVIEW.md",
)


class SecurityClaimContractTests(unittest.TestCase):
    def test_core_documentation_has_no_broken_local_links(self) -> None:
        for document in DOCUMENTS:
            content = document.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^]]+]\(([^)]+)\)", content):
                if "://" in target or target.startswith(("#", "mailto:")):
                    continue
                path = (document.parent / target.split("#", 1)[0]).resolve()
                self.assertTrue(path.exists(), f"broken link in {document}: {target}")

    def test_documented_security_posture_matches_runtime(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
        threat_model = (ROOT / "THREAT_MODEL.md").read_text(encoding="utf-8")

        for content in (readme, security, threat_model):
            normalized = " ".join(content.lower().split())
            self.assertRegex(normalized, r"secure-? aggregation")
            self.assertIn("formal differential privacy", normalized)
            self.assertIn("overlay", normalized)

        self.assertIn("centrally creates all", readme)
        self.assertIn("reads every review and label", threat_model)
        self.assertIn("resulting model parameters", threat_model)
        self.assertIn("per-client evaluation metrics", threat_model)
        secure_compose = (ROOT / "compose.secure.yaml").read_text(encoding="utf-8")
        self.assertIn("--ssl-ca-certfile", secure_compose)
        self.assertIn("--enable-supernode-auth", secure_compose)
        self.assertIn("--auth-supernode-private-key", secure_compose)

    def test_update_noise_is_not_documented_as_a_privacy_guarantee(self) -> None:
        for path in (
            ROOT / "README.md",
            ROOT / "SECURITY.md",
            ROOT / "THREAT_MODEL.md",
        ):
            content = " ".join(path.read_text(encoding="utf-8").split())
            self.assertIn("illustrative ablation, not formal differential", content)
            self.assertRegex(content, r"no (privacy )?accountant")
            self.assertIn("composition", content)
            self.assertIn("sensitivity", content)
            self.assertIn("epsilon", content)
            self.assertIn("delta", content)

        todo = (ROOT / "TODO.md").read_text(encoding="utf-8")
        for completed_control in (
            "Add TLS for Flower communication.",
            "Add SuperNode/client authentication and certificate lifecycle documentation.",
            "Add membership-inference and model-update leakage experiments.",
        ):
            self.assertIn(f"- [x] {completed_control}", todo)

        for completed_evaluation in (
            "Evaluate secure aggregation.",
            "If differential privacy is claimed, implement formal privacy accounting and publish epsilon/delta values.",
        ):
            self.assertIn(f"- [x] {completed_evaluation}", todo)

    def test_secure_aggregation_decision_does_not_claim_implementation(self) -> None:
        adr = " ".join(
            (ROOT / "docs" / "adr" / "0001-secure-aggregation.md")
            .read_text(encoding="utf-8")
            .split()
        )

        self.assertIn("Decision:** Defer implementation", adr)
        self.assertIn("Flower 1.32.1", adr)
        self.assertIn("Secure aggregation remains unimplemented", adr)
        self.assertIn("availability, not protocol validation", adr)
        self.assertIn("Huber aggregation is incompatible", adr)
        self.assertIn("num_examples` and metrics remain", adr)
        self.assertIn("No differential-privacy claim is made", adr)


if __name__ == "__main__":
    unittest.main()
