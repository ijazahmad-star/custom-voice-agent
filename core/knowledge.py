from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def setup_knowledge_base(config):
    print("Loading Knowledge Base...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    data = [
        # Office Logistics & Security
        "The secret project code for 'Project Zion' is 5543.",
        "The office is closed on Fridays, but the main lobby remains open for deliveries until 2 PM.",
        "The guest Wi-Fi password is 'CoffeeAndCode2026'.",
        "Emergency exits are located at the North and South ends of the 4th-floor corridor.",
        "The building security desk phone number is extension 9110.",

        # Human Resources & Benefits
        "The company holiday policy allows for 25 days of paid time off per year.",
        "Health insurance enrollment for 2026 ends on November 15th.",
        "Remote work is permitted on Mondays and Wednesdays for all engineering teams.",
        "The annual company retreat is scheduled for September 12th in Lake Tahoe.",

        # IT Support & Tools
        "To reset your Windows password, press Ctrl+Alt+Delete and select 'Change a password'.",
        "The IT support ticket portal can be reached at help.internal.ai.",
        "New laptop requests require manager approval and take approximately 5 business days to process.",
        "The office printer on the second floor is named 'Laser-Jet-02' and requires a badge swipe to print.",

        # Cafeteria & Perks
        "The company cafeteria serves pizza only on Thursdays and tacos on Tuesdays.",
        "Free snacks are replenished in the breakroom every Monday morning at 9 AM.",
        "Employees get a 20% discount at the 'Green Bean' coffee shop downstairs."
    ]
    vector_db = FAISS.from_texts(data, embeddings)
    return vector_db
