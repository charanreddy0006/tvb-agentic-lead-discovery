"""Step 7: safe email verification with syntax, DNS/MX, and source association checks."""

import re
from typing import Callable, Optional

import dns.resolver

from core.email_models import EmailCandidate, EmailVerificationResult, EmailVerificationStatus

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$")


class EmailVerificationService:
    """Performs non-intrusive validation; it never probes a mailbox by SMTP."""

    def __init__(self, mx_lookup: Optional[Callable[[str], bool]] = None) -> None:
        self.mx_lookup = mx_lookup or self._has_mx_record

    @staticmethod
    def _has_mx_record(domain: str) -> bool:
        try:
            return bool(dns.resolver.resolve(domain, "MX"))
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
            return False
        except dns.exception.DNSException:
            raise

    @staticmethod
    def _association_is_explicit(candidate: EmailCandidate) -> bool:
        return bool(candidate.email and candidate.evidence and candidate.founder_name and
                    candidate.email.lower() in candidate.evidence.lower() and
                    candidate.founder_name.lower() in candidate.evidence.lower())

    def verify(self, candidate: EmailCandidate) -> EmailVerificationResult:
        email = candidate.email
        if not email or not EMAIL_PATTERN.fullmatch(email):
            return EmailVerificationResult(email=email, status=EmailVerificationStatus.INVALID,
                                           reason="Email syntax is invalid.", evidence=candidate.evidence,
                                           verification_method="syntax")
        domain = email.rsplit("@", 1)[1]
        try:
            has_mx = self.mx_lookup(domain)
        except Exception:
            return EmailVerificationResult(email=email, status=EmailVerificationStatus.UNKNOWN,
                                           reason="Mail infrastructure could not be determined.", evidence=candidate.evidence,
                                           verification_method="syntax + DNS/MX")
        if not has_mx:
            return EmailVerificationResult(email=email, status=EmailVerificationStatus.INVALID,
                                           reason="Email domain has no valid MX mail infrastructure.", evidence=candidate.evidence,
                                           verification_method="syntax + DNS/MX")
        if not self._association_is_explicit(candidate):
            return EmailVerificationResult(email=email, status=EmailVerificationStatus.UNKNOWN,
                                           reason="Email has mail infrastructure but no explicit founder association.", evidence=candidate.evidence,
                                           verification_method="syntax + DNS/MX + source association")
        return EmailVerificationResult(email=email, status=EmailVerificationStatus.VERIFIED,
                                       reason="Syntax, MX infrastructure, and explicit founder association passed.", evidence=candidate.evidence,
                                       verification_method="syntax + DNS/MX + source association")
