from moneyforge.engine import Lead, audit_lead, rank_leads

def test_high_pain_lead_gets_actionable_offer():
    lead = Lead(
        business_name="Demo Shop",
        url="https://example.com",
        has_website=True,
        mobile_quality=30,
        cta_quality=20,
        offer_clarity=35,
        whatsapp_or_contact=40,
        localization=50,
        visible_ads=True,
    )
    audit = audit_lead(lead)
    assert audit.score >= 75
    assert audit.price_usd == 49
    assert audit.outreach_message

def test_ranked_descending():
    a = Lead(business_name="A", has_website=False)
    b = Lead(business_name="B", has_website=True, mobile_quality=90, cta_quality=90,
             offer_clarity=90, whatsapp_or_contact=90, localization=90)
    result = rank_leads([b, a])
    assert result[0]["lead"]["business_name"] == "A"
