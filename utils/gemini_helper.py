"""
═══════════════════════════════════════════════════════════════
Gemini AI Helper
═══════════════════════════════════════════════════════════════
Wraps the Google Gemini API for:
  - Waste explanation & disposal advice (explain_waste)
  - Conversational chat about waste topics (chat)

Uses the new google-genai SDK (v2.x).
Gracefully degrades when no API key is configured.
═══════════════════════════════════════════════════════════════
"""

# Lazy import so the server boots even if google-genai isn't installed
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class GeminiHelper:
    """Wrapper around the Google Gemini generative AI API (new SDK)."""

    MODEL_NAME = "gemini-2.5-flash"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key.strip()
        self.is_configured = False
        self._client = None

        if not GENAI_AVAILABLE:
            print("[WARN] google-genai not installed -- Gemini features disabled.")
            return

        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            print("[WARN] GEMINI_API_KEY not set -- running without Gemini explanations.")
            return

        try:
            self._client = genai.Client(api_key=self.api_key)
            self.is_configured = True
            print(f"[OK] Gemini AI configured ({self.MODEL_NAME})")
        except Exception as e:
            print(f"[WARN] Failed to configure Gemini: {e}")

    # ──────────────────────────────────────────────────────────
    # Public methods
    # ──────────────────────────────────────────────────────────

    def explain_waste(self, class_name: str, confidence: float) -> dict:
        """
        Generate an explanation and disposal advice for a classified waste item.

        Returns:
            {
              "explanation": str,
              "disposal": str,
              "tips": [str, ...],
              "source": "gemini" | "fallback"
            }
        """
        if not self.is_configured:
            return self._fallback_explanation(class_name)

        prompt = (
            f"You are a waste management expert. A waste classification AI identified "
            f"an item as '{class_name}' with {confidence:.1f}% confidence.\n\n"
            f"In 2-3 short sentences:\n"
            f"1. Briefly explain what this waste type is.\n"
            f"2. Give the best disposal method.\n"
            f"3. Mention one environmental impact if disposed incorrectly.\n\n"
            f"Then on a new line starting with 'DISPOSAL:', give a one-line disposal instruction.\n"
            f"Then on a new line starting with 'TIPS:', give 2-3 comma-separated quick tips."
        )

        try:
            response = self._client.models.generate_content(
                model=self.MODEL_NAME,
                contents=prompt,
            )

            text = response.text.strip()

            explanation = text
            disposal = ""
            tips = []

            # Parse structured output
            lines = text.splitlines()
            body_lines = []
            for line in lines:
                if line.startswith("DISPOSAL:"):
                    disposal = line.replace("DISPOSAL:", "").strip()
                elif line.startswith("TIPS:"):
                    raw_tips = line.replace("TIPS:", "").strip()
                    tips = [t.strip() for t in raw_tips.split(",") if t.strip()]
                else:
                    body_lines.append(line)

            explanation = "\n".join(body_lines).strip()

            return {
                "explanation": explanation,
                "disposal": disposal or "Follow local waste disposal guidelines.",
                "tips": tips,
                "source": "gemini",
            }

        except Exception as e:
            print(f"[WARN] Gemini explain_waste error: {e}")
            return self._fallback_explanation(class_name)

    def chat(self, user_message: str, context: dict = None, history: list = None) -> str:
        """
        Chat with Gemini about waste topics.
        """

        if not self.is_configured:
            return (
                "Gemini AI is not configured. Please set a valid GEMINI_API_KEY in your .env file."
            )

        system_ctx = (
            "You are WasteVision AI, a friendly waste management assistant. "
            "Help users dispose waste properly."
        )

        class_name = ""
        confidence = ""

        if context:
            class_name = context.get("class_name", "")
            confidence = context.get("confidence", "")

        if class_name:
            system_ctx += (
                f" The user scanned '{class_name}' with {confidence}% confidence. "
            )

        full_prompt = f"{system_ctx}\n\nUser: {user_message}"

        try:
            response = self._client.models.generate_content(
                model=self.MODEL_NAME,
                contents=full_prompt,
            )

            return response.text.strip()

        except Exception as e:
            print(f"[WARN] Gemini chat error: {e}")
            return f"Gemini Error: {str(e)}"

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_explanation(class_name: str) -> dict:
        """Return a rule-based explanation when Gemini is unavailable."""
        cl = class_name.lower()

        if any(k in cl for k in ["plastic", "glass", "metal", "paper", "cardboard"]):
            explanation = f"{class_name} is a recyclable material. Clean and dry items before placing in the recycling bin."
            disposal = "Place in the recycling bin after rinsing."
            tips = ["Rinse before recycling", "Remove caps/lids", "Flatten cardboard"]
        elif any(k in cl for k in ["organic", "food", "compost"]):
            explanation = f"{class_name} is organic waste that can be composted to enrich soil."
            disposal = "Place in compost bin or organic waste collection."
            tips = ["Compost at home", "Avoid composting meat/dairy", "Use a sealed bin"]
        elif any(k in cl for k in ["battery", "keyboard", "mobile", "pcb", "ewaste"]):
            explanation = f"{class_name} is e-waste containing hazardous materials. Never bin it."
            disposal = "Drop at a certified e-waste recycling centre."
            tips = ["Never bin e-waste", "Find local e-waste drives", "Check manufacturer take-back programs"]
        elif any(k in cl for k in ["hazard", "chemical", "medical", "toxic"]):
            explanation = f"{class_name} is hazardous waste requiring special disposal."
            disposal = "Contact local HazMat disposal services."
            tips = ["Do not pour down drains", "Store safely until disposal", "Use certified facilities"]
        else:
            explanation = f"{class_name} should be disposed of responsibly. Check your local waste authority guidelines."
            disposal = "Check local guidelines for proper disposal."
            tips = ["Reduce waste where possible", "Check if any parts are reusable", "Follow local rules"]

        return {
            "explanation": explanation,
            "disposal": disposal,
            "tips": tips,
            "source": "fallback",
        }
