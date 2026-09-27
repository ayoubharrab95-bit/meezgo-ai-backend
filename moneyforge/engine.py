from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List
from urllib.parse import urlparse

@dataclass
class Lead:
    business_name: str
    url: str = ""
    country: str = ""
    industry: str = ""
    has_website: bool = False
    mobile_quality: int = 0
    cta_quality: int = 0
    offer_clarity: int = 0
    whatsapp_or_contact: int = 0
    localization: int = 0
    visible_ads: bool = False
    notes: str = ""

@dataclass
class Audit:
    score: int
    pain_points: List[str]
    offer: str
    price_usd: int
    estimated_delivery_hours: float
    outreach_subject: str
    outreach_message: str

def clamp(n: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, int(n)))

def audit_lead(lead: Lead) -> Audit:
    pains: List[str] = []
    score = 0
    if not lead.has_website:
        pains.append("No dedicated website/landing page"); score += 24
    if lead.mobile_quality < 60:
        pains.append("Mobile experience needs improvement"); score += 18
    if lead.cta_quality < 60:
        pains.append("Weak or unclear conversion CTA"); score += 18
    if lead.offer_clarity < 60:
        pains.append("Offer/value proposition is unclear"); score += 14
    if lead.whatsapp_or_contact < 60:
        pains.append("Contact/order path is weak"); score += 12
    if lead.localization < 60:
        pains.append("Language/local-market experience can be improved"); score += 8
    if lead.visible_ads:
        pains.append("Active advertising increases the value of conversion fixes"); score += 10
    score = clamp(score)
    if score >= 75:
        price, hours, offer = 49, 2.5, "48-hour conversion-page rebuild"
    elif score >= 50:
        price, hours, offer = 29, 1.5, "Mobile + CTA conversion fix"
    elif score >= 30:
        price, hours, offer = 19, 1.0, "Quick conversion audit + fixes"
    else:
        price, hours, offer = 10, 0.5, "Mini conversion audit"
    domain = urlparse(lead.url).netloc if lead.url else lead.business_name
    subject = f"Quick conversion improvement for {lead.business_name}"
    top = "; ".join(pains[:3]) if pains else "a few conversion opportunities"
    message = (
        f"Hi, I reviewed {domain} and noticed {top}. "
        + "I can deliver a focused " + offer.lower() + " for $" + str(price) +
        ", with no long-term contract. The goal is simple: make it easier for visitors "
        "to understand the offer and contact/order. If useful, I can send a short before/after preview first."
    )
    return Audit(score, pains, offer, price, hours, subject, message)

def rank_leads(leads: List[Lead]) -> List[Dict]:
    ranked = []
    for lead in leads:
        ranked.append({"lead": asdict(lead), "audit": asdict(audit_lead(lead))})
    return sorted(ranked, key=lambda x: x["audit"]["score"], reverse=True)
