/**
 * SHAP-derived phishing keyword list — baked into the frontend for instant highlighting.
 *
 * HOW TO UPDATE:
 *   1. Run:  python3 generate_keywords.py         (in /backend)
 *   2. Copy the output keywords.json here as a JS export (format below).
 *   3. Rebuild the frontend:  npm run build
 *
 * Format: { word: { message: "...", shap_score: 0.0 }, ... }
 * Only the keys (words) are used for live highlighting in the editor.
 * The messages are used for the recommendations panel.
 */

const KEYWORDS = {
  "verify":        { message: "Requests to verify identity are a classic phishing tactic.",             shap_score: 0.0 },
  "password":      { message: "Never send or request passwords through email.",                         shap_score: 0.0 },
  "login":         { message: "Avoid clicking login links in emails — go directly to the official site.", shap_score: 0.0 },
  "bank":          { message: "Impersonating banks is one of the most common phishing techniques.",     shap_score: 0.0 },
  "urgent":        { message: "Artificial urgency pressures users into acting without thinking.",       shap_score: 0.0 },
  "click":         { message: "Legitimate organizations rarely ask you to click unverified links.",     shap_score: 0.0 },
  "account":       { message: "Threats about account suspension are used to create panic.",             shap_score: 0.0 },
  "suspended":     { message: "Account suspension threats are used to trick users into panic actions.", shap_score: 0.0 },
  "winner":        { message: "Prize and lottery announcements are almost always scams.",               shap_score: 0.0 },
  "free":          { message: "Unsolicited free offers are a red flag for phishing.",                   shap_score: 0.0 },
  "confirm":       { message: "Requests to confirm personal details are a phishing red flag.",          shap_score: 0.0 },
  "update":        { message: "Forced update requests via email are suspicious.",                       shap_score: 0.0 },
  "security":      { message: "Fake security alerts are used to scare users into acting fast.",        shap_score: 0.0 },
  "prize":         { message: "Lottery/prize claims are common phishing tactics.",                      shap_score: 0.0 },
  "httplink":      { message: "Email contains an insecure HTTP link (no SSL) — legitimate emails use HTTPS.", shap_score: 0.0 },
  "credit":        { message: "Legitimate services never ask for credit card details via email.",       shap_score: 0.0 },
  "payment":       { message: "Unexpected payment requests should always be verified through official channels.", shap_score: 0.0 },
  "invoice":       { message: "Fake invoices are commonly used to trick users into opening attachments.", shap_score: 0.0 },
  "expire":        { message: "Expiry warnings are used to create urgency and pressure quick action.", shap_score: 0.0 },
  "authorize":     { message: "Requests to authorize actions via email links are suspicious.",          shap_score: 0.0 },
  "validate":      { message: "Requests to validate identity are a classic phishing tactic.",           shap_score: 0.0 },
  "access":        { message: "Fake account recovery emails are used to steal credentials.",            shap_score: 0.0 },
  "claim":         { message: "Claim requests are commonly used in prize scam phishing emails.",        shap_score: 0.0 },
  "reward":        { message: "Fake reward offers are a common phishing lure.",                         shap_score: 0.0 },
  "transfer":      { message: "Wire transfer requests via email are a major fraud red flag.",           shap_score: 0.0 },
  "alert":         { message: "Fake security alerts are designed to cause panic and rushed decisions.", shap_score: 0.0 },
  "ssn":           { message: "No legitimate organization requests your Social Security Number via email.", shap_score: 0.0 },
  "limited":       { message: "'Limited time' language is a pressure tactic used in phishing.",         shap_score: 0.0 },
  "notify":        { message: "Unsolicited notifications asking for action should be treated with caution.", shap_score: 0.0 },
  "protect":       { message: "Fake protection warnings trick users into clicking malicious links.",    shap_score: 0.0 },
};

// ── Derived exports used by Scanner.jsx ──────────────────────────────────────

/** Array of keyword strings — used for live yellow highlighting in the editor. */
export const PHISHING_KEYWORDS = Object.keys(KEYWORDS);

/** Full keyword map — used for recommendation messages in the tips panel. */
export const PHISHING_KEYWORDS_MAP = KEYWORDS;
