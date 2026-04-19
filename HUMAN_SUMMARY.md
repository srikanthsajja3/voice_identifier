# 🌆 Urban Noise Intelligence: Beyond Just Listening

### Why this project matters
Most AI systems are like "Black Boxes"—they give you an answer, but they never tell you *how* they got there. In a city, if an AI marks a sound as a "Gunshot," we need to be 100% sure it didn't just mistake a car backfire for a weapon. 

This project isn't just about classifying noise; it's about **trust**. By using **Explainable AI (XAI)**, we allow the machine to show us its "thought process" through visual heatmaps.

---

### What does it actually do?
Imagine you’re walking through a busy street. Your ears are bombarded by a mix of sirens, construction, and chatter. This system does three things:
1.  **Listens:** It captures 4 seconds of audio.
2.  **Visualizes:** It turns that sound into a "picture" (a Spectrogram) so the AI can "see" the noise patterns.
3.  **Explains:** It highlights the exact parts of that picture that led to its decision. If it says "Siren," it will highlight the rhythmic waving frequencies typical of emergency vehicles.

---

### Key Features (The "Human" Version)
*   **The "Attention" Heatmap:** We use a technique called Grad-CAM. Think of it as a highlighter that the AI uses to show you exactly which part of the sound it was "looking" at.
*   **Digital Fingerprints (MFCCs):** Sound has texture. We extract the "DNA" of the sound to tell the difference between a dog bark and a child playing, even if they happen at the same volume.
*   **Real-time Dashboard:** No complicated code needed to run it. Just press "Record" in your browser, and the AI starts talking to you.
*   **Smart City Vision:** We’ve added a trend estimator. This helps city planners realize, for example, "Hey, construction noise is up 12% this month, maybe we should adjust the traffic flow."

---

### How to see it in action
1.  **Activate your environment:** `source venv/bin/activate`
2.  **Start the app:** `python3 app.py`
3.  **Open your browser:** Go to `http://127.0.0.1:8080`
4.  **Try it:** Record a whistle or play a siren sound from your phone near the mic. Watch the AI explain its decision to you in real-time.

---

### Final Thoughts
This project bridges the gap between complex Deep Learning and real-world transparency. It’s a tool built for people—planners, safety officers, and citizens—who want to understand their environment better.
