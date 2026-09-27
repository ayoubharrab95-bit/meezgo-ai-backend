import unittest
from moneyforge.engine import Lead, audit_lead, rank_leads

class MoneyForgeTests(unittest.TestCase):
    def test_high_pain_lead_gets_actionable_offer(self):
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
        self.assertGreaterEqual(audit.score, 75)
        self.assertEqual(audit.price_usd, 49)
        self.assertTrue(audit.outreach_message)

    def test_ranked_descending(self):
        a = Lead(business_name="A", has_website=False)
        b = Lead(
            business_name="B",
            has_website=True,
            mobile_quality=90,
            cta_quality=90,
            offer_clarity=90,
            whatsapp_or_contact=90,
            localization=90,
        )
        result = rank_leads([b, a])
        self.assertEqual(result[0]["lead"]["business_name"], "A")

if __name__ == "__main__":
    unittest.main()
