import pandas as pd
import re

def classify_with_reasoning(text):
    text = str(text).lower()
    
    # Label keywords
    complaint_keywords = [
        r"\bcomplain\w*\b", r"\bissue\b", r"\bnot work\w*\b", 
        r"\bfault\b", r"\bpoor\b", r"\bbad\b", r"\bunsatisfactory\b", r"\bunhappy\b", 
        r"\bdissatisfied\b", r"\bproblem\b", r"\berror\b", r"\bglitch\b"
    ]
    it_service_keywords = [
        r"\bit\b", r"\bservice\b", r"\bsupport\b", r"\btechnical\b", r"\bconfiguration\b", 
        r"\binstallation\b", r"\bsetup\b", r"\bconnection\b", r"\bsystem\b", r"\bnetwork\b", r"\bsoftware\b"
    ]
    cloud_autoscaling_keywords = [
        r"\bcloud\b", r"\bauto-?scaling\b", r"\bscal\w*\b", r"\bdynamic\b", 
        r"\bresources\b", r"\bvirtual\b", r"\bserver\b", r"\bcapacity\b", r"\badjust\b"
    ]
    under_provisioning_keywords = [
        r"\bunder-?provision\w*\b", r"\blow storage\b", r"\blow space\b", r"\bhdd\b", 
        r"\bdrive\b", r"\binsufficient storage\b", r"\binsufficient space\b", 
        r"\bstorage\b", r"\bdisk\b", r"\bnot enough\b", r"\bcapacity\b"
    ]
    
    # Check for each label
    if any(re.search(keyword, text) for keyword in complaint_keywords):
        return "complaint"
    elif any(re.search(keyword, text) for keyword in it_service_keywords):
        return "IT service"
    elif any(re.search(keyword, text) for keyword in cloud_autoscaling_keywords):
        return "cloud autoscaling"
    elif any(re.search(keyword, text) for keyword in under_provisioning_keywords):
        return "under-provisioning of HDD"
    else:
        # Default 
        return "IT service"

app_gallery_df = pd.read_csv('data/preprocessed_appgallery_data.csv')
purchasing_df = pd.read_csv('data/preprocessed_purchasing_data.csv')

# Apply classification
app_gallery_df['label'] = app_gallery_df['Interaction content'].apply(classify_with_reasoning)
purchasing_df['label'] = purchasing_df['Interaction content'].apply(classify_with_reasoning)

app_gallery_df.to_csv('data/AppGallery_labelled.csv', index=False)
purchasing_df.to_csv('data/Purchasing_labelled.csv', index=False)