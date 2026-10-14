"""Comprehensive retrieval benchmark evaluating TurboQuantIndex against full-precision float32 search.

Evaluates 2-bit polar quantization (with and without QJL correction) across candidate pool sizes
K in [10, 25, 50, 100] on Indic multilingual text queries and corpus documents.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from turboquant.embed import extract_indic_embeddings
from turboquant.index import TurboQuantIndex


class IndicFallbackTokenizer:
    """Deterministic tokenization fallback for Indic multilingual text without remote network access."""

    def __init__(self, vocab_size: int = 5000, pad_token_id: int = 0) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        texts: Sequence[str],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        tokenized_seqs: List[List[int]] = []
        for text in texts:
            words = text.strip().split()
            # Deterministic word & character hashing to token IDs in [10, vocab_size - 1]
            seq = [abs(hash(w)) % (self.vocab_size - 10) + 10 for w in words] or [10]
            if truncation and max_length is not None:
                seq = seq[:max_length]
            tokenized_seqs.append(seq)

        max_len = max((len(s) for s in tokenized_seqs), default=0) if padding else 0

        input_ids = []
        attention_mask = []
        for seq in tokenized_seqs:
            seq_len = len(seq)
            if padding and max_len > seq_len:
                padded_seq = seq + [self.pad_token_id] * (max_len - seq_len)
                mask = [1] * seq_len + [0] * (max_len - seq_len)
            else:
                padded_seq = seq
                mask = [1] * seq_len
            input_ids.append(padded_seq)
            attention_mask.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}  # type: ignore[return-value]


def load_indic_bert_model(
    model_name: str = "ai4bharat/indic-bert",
) -> Tuple[Any, Any, str]:
    """Load IndicBERT model and tokenizer, falling back cleanly to an in-memory 768-dim model if offline.

    Returns:
        Tuple of (model, tokenizer, model_source_description).
    """
    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        return model, tokenizer, f"ai4bharat/indic-bert (online HuggingFace)"
    except Exception as exc:
        from transformers import BertConfig, BertModel

        # IndicBERT hidden dimension is 768
        config = BertConfig(
            vocab_size=5000,
            hidden_size=768,
            num_attention_heads=6,
            num_hidden_layers=4,
            intermediate_size=1024,
        )
        torch.manual_seed(42)
        model = BertModel(config)
        model.eval()
        tokenizer = IndicFallbackTokenizer(vocab_size=5000)
        return (
            model,
            tokenizer,
            f"IndicBERT architecture fallback (768-dim, offline/gated fallback: {type(exc).__name__})",
        )


def build_indic_evaluation_dataset() -> Tuple[List[str], List[str], List[str]]:
    """Build a comprehensive Indic multilingual evaluation set with parallel and thematic sentence pairs.

    Returns:
        Tuple of (queries, corpus, language_labels).
    """
    # Multilingual parallel sentence pairs across Indic languages: Hindi, Bengali, Tamil, Telugu,
    # Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia.
    seed_parallel_pairs: List[Tuple[str, str, str]] = [
        # (Query, Matching Corpus Document, Language)
        ("भारत की राजधानी क्या है?", "नई दिल्ली भारत की आधिकारिक राजधानी और प्रशासनिक केंद्र है।", "Hindi"),
        ("हिमालय पर्वत कहाँ स्थित है?", "हिमालय पर्वतमाला भारत के उत्तरी भाग में भव्य रूप से फैली हुई है।", "Hindi"),
        ("भारतीय अर्थव्यवस्था का मुख्य आधार क्या है?", "कृषि और सेवा क्षेत्र भारतीय अर्थव्यवस्था के प्रमुख स्तंभ हैं।", "Hindi"),
        ("भारत का स्वतंत्रता दिवस कब मनाया जाता है?", "भारत प्रत्येक वर्ष १५ अगस्त को अपना स्वतंत्रता दिवस मनाता है।", "Hindi"),
        ("गंगा नदी कहाँ से निकलती है?", "गंगा नदी उत्तराखंड में गंगोत्री हिमनद से निकलकर बंगाल की खाड़ी में गिरती है।", "Hindi"),
        ("ভারতের রাজধানী কোথায়?", "নতুন দিল্লি ভারতের সরকারি রাজধানী এবং প্রশাসনিক কেন্দ্র।", "Bengali"),
        ("হিমালয় পর্বত কোথায় অবস্থিত?", "হিমালয় পর্বতমালা ভারতের উত্তরাঞ্চলে বিস্তীর্ণভাবে অবস্থান করছে।", "Bengali"),
        ("রবীন্দ্রনাথ ঠাকুর কে ছিলেন?", "রবীন্দ্রনাথ ঠাকুর ছিলেন বিশ্বখ্যাত বাঙালি কবি ও নোবেল পুরস্কার বিজয়ী।", "Bengali"),
        ("সুন্দরবন কোথায় অবস্থিত?", "সুন্দরবন পশ্চিমবঙ্গ ও বাংলাদেশের দক্ষিণে অবস্থিত বিশ্বের বৃহত্তম ম্যানগ্রোভ বন।", "Bengali"),
        ("কলকাতার বিখ্যাত মিষ্টি কোনটি?", "রসগোল্লা এবং সন্দেশ কলকাতার অত্যন্ত জনপ্রিয় এবং ঐতিহ্যবাহী মিষ্টি।", "Bengali"),
        ("இந்தியாவின் தலைநகரம் எது?", "புது தில்லி இந்தியாவின் அதிகாரப்பூர்வ தலைநகரமாகவும் மையமாகவும் உள்ளது.", "Tamil"),
        ("தமிழ்நாட்டின் தலைநகரம் எது?", "சென்னை தமிழ்நாட்டின் தலைநகரமாகவும் முக்கிய துறைமுக நகரமாகவும் விளங்குகிறது.", "Tamil"),
        ("திருக்குறளை இயற்றியவர் யார்?", "திருவள்ளுவர் உலகப் புகழ்பெற்ற திருக்குறள் நீதிநூலை இயற்றினார்.", "Tamil"),
        ("காவிரி நதி எங்கு பாய்கிறது?", "காவிரி நதி கர்நாடகம் மற்றும் தமிழ்நாடு வழியாக பாய்ந்து வங்காள விரிகுடாவில் கலக்கிறது.", "Tamil"),
        ("பொங்கல் திருநாள் எப்போது கொண்டாடப்படுகிறது?", "தை மாதத் தொடக்கத்தில் தமிழர் திருநாளான பொங்கல் பெருவிழா கொண்டாடப்படுகிறது.", "Tamil"),
        ("భారతదేశ రాజధాని ఏది?", "న్యూఢిల్లీ భారతదేశ అధికారిక రాజధాని మరియు పరిపాలనా కేంద్రం.", "Telugu"),
        ("తెలంగాణ రాజధాని ఏది?", "హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని మరియు ప్రముఖ ఐటీ కేంద్రం.", "Telugu"),
        ("గోదావరి నది ఎక్కడ పుట్టింది?", "గోదావరి నది మహారాష్ట్రలోని నాసిక్ వద్ద పుట్టి బంగాళాఖాతంలో కలుస్తుంది.", "Telugu"),
        ("ఆంధ్రప్రదేశ్ ముఖ్య నృత్యం ఏది?", "కూచిపూడి ఆంధ్రప్రదేశ్ రాష్ట్రానికి చెందిన సాంప్రదాయ శాస్త్రీయ నృత్యం.", "Telugu"),
        ("భారత రాజ్యాంగ నిర్మాత ఎవరు?", "డాక్టర్ బి. ఆర్. అంబేద్కర్ భారత రాజ్యాంగ రూపకర్తగా ప్రసిద్ధి చెందారు.", "Telugu"),
        ("महाराष्ट्राची राजधानी कोणती आहे?", "मुंबई ही महाराष्ट्राची राजधानी आणि भारताची आर्थिक राजधानी मानली जाते.", "Marathi"),
        ("शिवाजी महाराज कोण होते?", "छत्रपती शिवाजी महाराज हे मराठा साम्राज्याचे संस्थापक आणि थोर योद्धा होते.", "Marathi"),
        ("महाराष्ट्रातील प्रसिद्ध सण कोणता?", "गणेशोत्सव हा संपूर्ण महाराष्ट्रात अत्यंत उत्साहाने आणि भक्तीने साजरा केला जातो.", "Marathi"),
        ("गोदावरी नदीचे उगमस्थान कोठे आहे?", "गोदावरी नदीचा उगम महाराष्ट्रातील नाशिक जवळील त्र्यंबकेश्वर येथे होतो.", "Marathi"),
        ("ગુજરાતનું પાટનગર કયું છે?", "ગાંધીનગર ગુજરાત રાજ્યનું પાટનગર અને વહીવટી વડું મથક છે.", "Gujarati"),
        ("અમદાવાદ શાના માટે પ્રખ્યાત છે?", "અમદાવાદ કાપડ ઉદ્યોગ અને ઐતિહાસિક સાબરમતી આશ્રમ માટે જાણીતું છે.", "Gujarati"),
        ("ગીર રાષ્ટ્રીય ઉદ્યાન ક્યાં આવેલું છે?", "ગીર રાષ્ટ્રીય ઉદ્યાન ગુજરાતમાં એશિયાટિક સિંહોનું એકમાત્ર કુદરતી રહેઠાણ છે.", "Gujarati"),
        ("ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?", "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಮತ್ತು ಭಾರತದ ಸಿಲಿಕಾನ್ ವ್ಯಾಲಿ ಎಂದು ಕರೆಯಲ್ಪಡುತ್ತದೆ.", "Kannada"),
        ("ಹಂಪಿ ಎಲ್ಲಿದೆ?", "ಹಂಪಿ ಕರ್ನಾಟಕದ ಬಳ್ಳಾರಿ ಜಿಲ್ಲೆಯಲ್ಲಿರುವ ವಿಜಯನಗರ ಸಾಮ್ರಾಜ್ಯದ ಐತಿಹಾಸಿಕ ತಾಣವಾಗಿದೆ.", "Kannada"),
        ("ಕಾವೇರಿ ನದಿಯ ಉಗಮ ಎಲ್ಲಿ?", "ಕಾವೇರಿ ನದಿಯು ಕೊಡಗು ಜಿಲ್ಲೆಯ ತಲಕಾವೇರಿಯಲ್ಲಿ ಉಗಮಿಸುತ್ತದೆ.", "Kannada"),
        ("കേരളത്തിന്റെ തലസ്ഥാനം ഏതാണ്?", "തിരുവനന്തപുരം കേരളത്തിന്റെ തലസ്ഥാന നഗരവും പ്രധാന സാംസ്കാരിക കേന്ദ്രവുമാണ്.", "Malayalam"),
        ("കേരളത്തിന്റെ പ്രധാന ഉത്സവം ഏതാണ്?", "ഓണം കേരളീയരുടെ ദേശീയോത്സവവും കാർഷിക സമൃദ്ധിയുടെ ആഘോഷവുമാണ്.", "Malayalam"),
        ("കൊച്ചി എന്തിനാണ് പ്രസിദ്ധം?", "കൊച്ചി അറബിക്കടലിന്റെ റാണി എന്നറിയപ്പെടുന്ന പ്രകൃതിദത്ത തുറമുഖ നഗരമാണ്.", "Malayalam"),
        ("ਪੰਜਾਬ ਦੀ ਰਾਜਧਾਨੀ ਕਿਹੜੀ ਹੈ?", "ਚੰਡੀਗੜ੍ਹ ਪੰਜਾਬ ਅਤੇ ਹਰਿਆਣਾ ਦੋਵਾਂ ਰਾਜਾਂ ਦੀ ਸਾਂਝੀ ਰਾਜਧਾਨੀ ਹੈ।", "Punjabi"),
        ("ਅੰਮ੍ਰਿਤਸਰ ਦਾ ਮਸ਼ਹੂਰ ਅਸਥਾਨ ਕਿਹੜਾ ਹੈ?", "ਸ੍ਰੀ ਹਰਿਮੰਦਰ ਸਾਹਿਬ (ਗੋਲਡਨ ਟੈਂਪਲ) ਅੰਮ੍ਰਿਤਸਰ ਦਾ ਵਿਸ਼ਵ ਪ੍ਰਸਿੱਧ ਧਾਰਮਿਕ ਅਸਥਾਨ ਹੈ।", "Punjabi"),
        ("ਵਿਸਾਖੀ ਦਾ ਤਿਉਹਾਰ ਕਦੋਂ ਮਨਾਇਆ ਜਾਂਦਾ ਹੈ?", "ਵਿਸਾਖੀ ਦਾ ਤਿਉਹਾਰ ਹਰ ਸਾਲ ਅਪ੍ਰੈਲ ਮਹੀਨੇ ਵਿੱਚ ਕਣਕ ਦੀ ਵਾਢੀ ਵੇਲੇ ਮਨਾਇਆ ਜਾਂਦਾ ਹੈ।", "Punjabi"),
        ("ଓଡ଼ିଶାର ରାଜଧାନୀ କେଉଁଟି?", "ଭୁବନେଶ୍ୱର ଓଡ଼ିଶାର ରାଜଧାନୀ ଏବଂ ଏହାକୁ ମନ୍ଦିରମାଳିନୀ ନଗରୀ କୁହାଯାଏ।", "Odia"),
        ("ପୁରୀ ଜଗନ୍ନାଥ ମନ୍ଦିର କେଉଁଠାରେ ଅବସ୍ଥିତ?", "ପୁରୀ ଜଗନ୍ନାଥ ମନ୍ଦିର ଓଡ଼ିଶାର ସମୁଦ୍ର କୂଳରେ ଅବସ୍ଥିତ ଏକ ପବିତ୍ର ତୀର୍ଥସ୍ଥଳ।", "Odia"),
        ("କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିର କାହିଁକି ପ୍ରସିଦ୍ଧ?", "କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିର ଏହାର ଅଦ୍ଭୁତ କଳା ସ୍ଥାପତ୍ୟ ଏବଂ ରଥ ସଦୃଶ ଗଠନ ପାଇଁ ପ୍ରସିଦ୍ଧ।", "Odia"),
        ("مصر کی قدیم تہذیب کس دریا پر تھی؟", "وادی نیل کی قدیم تہذیب دریائے نیل کے زرخیز کناروں پر پروان چڑھی تھی۔", "Urdu"),
        ("اردو زبان کا آغاز کہاں ہوا؟", "اردو زبان کا آغاز برصغير پاک و ہند کے مختلف تہذیبی اور لسانی سنگم سے ہوا۔", "Urdu"),
        ("తాజ్ మహల్ ఎక్కడ ఉంది?", "తాజ్ మహల్ భారతదేశంలోని ఉత్తర ప్రదేశ్ రాష్ట్రంలోని ఆగ్రా నగరంలో యమునా నది తీరాన ఉంది.", "Telugu"),
        ("ताजमहल कहाँ स्थित है?", "ताजमहल उत्तर प्रदेश के आगरा शहर में यमुना नदी के तट पर स्थित विश्व प्रसिद्ध स्मारक है।", "Hindi"),
        ("সৌরজগতের বৃহত্তম গ্রহ কোনটি?", "বৃহস্পতি সৌরজগতের বৃহত্তম গ্যাসীয় গ্রহ হিসেবে পরিচিত।", "Bengali"),
        ("सौरमंडल का सबसे बड़ा ग्रह कौन सा है?", "बृहस्पति हमारे सौरमंडल का सबसे विशाल और भारी ग्रह है।", "Hindi"),
        ("சூரிய குடும்பத்தில் மிகப்பெரிய கிரகம் எது?", "வியாழன் நமது சூரிய குடும்பத்தின் மிகப்பெரிய கோளாக திகழ்கிறது.", "Tamil"),
        ("కంప్యూటర్ పితామహుడు ఎవరు?", "చార్లెస్ బాబేజ్ కంప్యూటర్ పితామహుడిగా ప్రపంచవ్యాప్తంగా గుర్తింపు పొందారు.", "Telugu"),
        ("संगणक का जनक किसे कहा जाता है?", "चार्ल्स बैबेज को आधुनिक कंप्यूटर का जनक माना जाता है।", "Hindi"),
        ("గూగుల్ సంస్థ ఎప్పుడు స్థాపించబడింది?", "లారీ పేజ్ మరియు సెర్గీ బ్రిన్ 1998లో గూగుల్ సెర్చ్ ఇంజిన్ సంస్థను ప్రారంభించారు.", "Telugu"),
        ("गूगल की स्थापना कब हुई थी?", "गूगल की स्थापना 1998 में लैरी पेज और सर्गेई ब्रिन द्वारा की गई थी।", "Hindi"),
    ]

    # Additional diverse Indic corpus documents across multiple domains (Science, Geography, Arts, History)
    distractor_documents: List[str] = [
        "आर्टिफिशियल इंटेलिजेंस और मशीन लर्निंग आधुनिक तकनीकी क्रांति के मुख्य आधार हैं।",
        "कृत्रिम बुद्धिमत्ता भविष्य की गणना तकनीकों को पूरी तरह बदल रही है।",
        "தமிழ் மொழி திராவிட மொழிக் குடும்பத்தின் மிகத் தொன்மையான செம்மொழியாகும்.",
        "சங்க இலக்கியங்கள் பண்டைத் தமிழர்களின் வீரம், காதல் மற்றும் பண்பாட்டை விவரிக்கின்றன.",
        "భారతీయ అంతరిక్ష పరిశోధనా సంస్థ (ఇస్రో) ఉపగ్రహ ప్రయోగాలలో అద్భుత విజయాలు సాధించింది.",
        "చంద్రయాన్ మరియు ఆదిత్య ఎల్-1 ప్రయోగాలు భారతదేశ శాస్త్ర సాంకేతిక సత్తాను నిరూపించాయి.",
        "বাংলা সাহিত্য রবীন্দ্রনাথ ঠাকুর, কাজী নজরুল ইসলাম மற்றும் বঙ্কিমচন্দ্রের সৃষ্টিতে সমৃদ্ধ।",
        "সত্যজিৎ রায় ভারতীয় চলচ্চিত্রকে আন্তর্জাতিক অঙ্গনে সর্বোচ্চ সম্মানের আসনে অধিষ্ঠিত করেছিলেন।",
        "छत्रपती संभाजी महाराजांनी मराठा साम्राज्याचे स्वातंत्र्य राखण्यासाठी मोठा लढा दिला.",
        "अजिंठा आणि वेरूळ येथील लेणी प्राचीन भारतीय शिल्पकलेचे उत्कृष्ट नमुने आहेत.",
        "ગુજરાતમાં આવેલું સ્ટેચ્યુ ઓફ યુનિટી વિશ્વની સૌથી ઊંચી પ્રતિમા છે.",
        "નર્મદા નદી ગુજરાત અને મધ્યપ્રદેશની જીવાદોરી સમાન મહત્વની નદી છે.",
        "ಯಕ್ಷಗಾನ ಕರ್ನಾಟಕದ ಕರಾವಳಿ ತೀರದ ಅದ್ಭುತ ಜಾನಪದ ನೃತ್ಯ ನಾಟಕ ಕಲೆಯಾಗಿದೆ.",
        "ಕುವೆಂಪು ಕನ್ನಡ ಸಾಹಿತ್ಯ ಲೋಕದ ಪ್ರಸಿದ್ಧ ರಾಷ್ಟ್ರಕವಿ ಮತ್ತು ಜ್ಞಾನಪೀಠ ಪ್ರಶಸ್ತಿ ಪುರಸ್ಕೃತರು.",
        "കഥകളി കേരളത്തിന്റെ പരമ്പരാഗതമായ സമ്പന്ന വേഷവിധാനങ്ങളുള്ള ശാസ്ത്രീയ നൃത്തരൂപമാണ്.",
        "ആയുർവേദം ഭാരതത്തിന്റെ പൗരാണിക ചികിത്സാ സമ്പ്രദായവും ജീവിതരീതിയുമാണ്.",
        "ਭੰਗੜਾ ਅਤੇ ਗਿੱਧਾ ਪੰਜਾਬ ਦੇ ਜੋਸ਼ੀਲੇ ਅਤੇ ਰਵਾਇਤੀ ਲੋਕ ਨਾਚ ਹਨ।",
        "ਪੰਜ ਦਰਿਆਵਾਂ ਦੀ ਧਰਤੀ ਹੋਣ ਕਰਕੇ ਇਸ ਖਿੱਤੇ ਦਾ ਨਾਂ ਪੰਜਾਬ ਪਿਆ।",
        "ଓଡ଼ିଶୀ ନୃତ୍ୟ ଭାରତର ଆଠଟି ପ୍ରମୁଖ ଶାସ୍ତ୍ରୀୟ ନୃତ୍ୟ ମଧ୍ୟରୁ ଅନ୍ୟତମ।",
        "ଚିଲିକା ହ୍ରଦ ଭାରତର ସର୍ବବୃହତ ଲୁଣିଆ ପାଣିର ହ୍ରଦ ଏବଂ ପକ୍ଷୀମାନଙ୍କ ଆଶ୍ରୟସ୍ଥଳ।",
        "गौतम बुद्ध ने अपना पहला उपदेश सारनाथ में धर्मचक्रप्रवर्तन के रूप में दिया था।",
        "अशोक के शिलालेख भारत के विभिन्न कोनों में ब्राह्मी और खरोष्ठी लिपि में पाए गए हैं।",
        "சிந்து சமவெளி நாகரிகம் பழங்கால நாகரிகங்களில் மிகவும் திட்டமிடப்பட்ட நகர அமைப்பைக் கொண்டிருந்தது.",
        "மெகஞ்சதாரோ மற்றும் ஹரப்பா நகரங்கள் பண்டைய வடிகால் அமைப்பிற்கு சான்றாக உள்ளன.",
        "మౌర్య సామ్రాజ్యం చంద్రగుప్త మౌర్యుని నేతృత్వంలో భారతదేశంలో విస్తరించింది.",
        "విక్రమార్కుని నవరత్నాలలో కాళిదాసు గొప్ప సంస్కృత కవిగా కొనియాడబడ్డాడు.",
        "सौर ऊर्जा और पवन ऊर्जा हरित ऊर्जा के मुख्य स्रोत हैं जो प्रदूषण कम करते हैं।",
        "জলবায়ু পরিবর্তন মোকাবেলায় নবায়নযোগ্য শক্তির ব্যবহার দ্রুত বৃদ্ধি পাচ্ছে।",
        "பசுமை இல்ல வாயுக்கள் புவி வெப்பமயமாதலுக்கு முதன்மைக் காரணமாக அமைகின்றன.",
        "పర్యావరణ పరిరక్షణ కోసం ప్లాస్టిక్ వాడకాన్ని తగ్గించడం అత్యంత ఆవశ్యకం.",
        "भारतीय रेल नेटवर्क दुनिया के सबसे बड़े रेल नेटवर्कों में से एक है।",
        "भारतीय अंतरिक्ष एजेंसी इसरो ने एक ही मिशन में 104 उपग्रह प्रक्षेपित कर विश्व रिकॉर्ड बनाया था।",
        "மங்கள்யான் திட்டம் இந்தியாவின் முதல் செவ்வாய் கிரக ஆய்வுப் பயணமாகும்.",
        "ക്രിക്കറ്റ് ഇന്ത്യയിലെ ഏറ്റവും ജനപ്രിയമായ കായിക വിനോദമാണ്.",
        "സച്ചിൻ തെണ്ടുൽക്കർ അന്താരാഷ്ട്ര ക്രിക്കറ്റിൽ നൂറ് സെഞ്ചുറികൾ തികച്ച ഏക താരമാണ്.",
        "हॉकी भारत का पारंपरिक राष्ट्रीय खेल है जिसमें देश ने कई ओलंपिक स्वर्ण पदक जीते।",
        "నీలి విప్లవం ద్వారా చేపల పెంపకం మరియు సముద్ర ఉత్పత్తుల ఎగుమతి గణనీયంగా పెరిగింది.",
        "শ্বেত বিপ্লবের জনক ড. ভার্গিস কুরিয়েন ভারতে দুগ্ধ উৎপাদনে বিপ্লব ঘটিয়েছিলেন।",
        "ഹരിത വിപ്ലവം ഇന്ത്യയിൽ ഭക്ഷ്യധാന്യ ഉൽപാദനത്തിൽ സ്വയംപര്യാപ്തത കൈവരിക്കാൻ സഹായിച്ചു.",
        "सूक्ष्म अर्थशास्त्र व्यक्तिगत उपभोक्ताओं और कंपनियों के आर्थिक निर्णयों का अध्ययन करता है।",
        "ಮ್ಯಾಕ್ರೋ ಅರ್ಥಶಾಸ್ತ್ರವು ರಾಷ್ಟ್ರೀಯ ಆದಾಯ, ಹಣದುಬ್ಬರ ಮತ್ತು ನಿರುದ್ಯೋಗವನ್ನು ವಿಶ್ಲೇಷಿಸುತ್ತದೆ.",
        "ಕ್ವಾಂಟಮ್ ಕಂಪ್ಯೂಟಿಂಗ್ ಸಾಂಪ್ರದಾಯಿಕ ಕಂಪ್ಯೂಟರ್‌ಗಳಿಗಿಂತ ಲಕ್ಷಾಂತರ ಪಟ್ಟು ವೇಗವಾಗಿ ಸಂಕೀರ್ಣ ಲೆಕ್ಕಾಚಾರಗಳನ್ನು ಮಾಡುತ್ತದೆ.",
        "சூப்பர் கண்டக்டர்கள் மின்சாரத்தை எந்தவித எதிர்ப்பும் இன்றி கடத்தும் திறன் கொண்டவை.",
        "ডিএনএ এবং আরএনএ জীবের জিনগত তথ্য সংরক্ষণ এবং স্থানান্তরের মূল বাহক।",
        "రక్తంలో ఎర్ర రక్తకణాలు శరీరంలోని అన్ని భాగాలకు ఆక్సిజన్‌ను మోసుకెళ్తాయి.",
        "डीएनए की डबल हेलिक्स संरचना की खोज वाटसन और क्रिक ने की थी।",
        "न्यूरॉन्स तंत्रिका तंत्र की मौलिक कार्यात्मक इकाइयां हैं।",
        "மস্তিষ্কের செல்கள் ஒன்றுடன் ஒன்று சினாப்ஸ் மூலம் தகவல்களைப் பரிமாறுகின்றன.",
        "ஒளிச்சேர்க்கை மூலம் தாவரங்கள் சூரிய ஒளியைப் பயன்படுத்தி குளுக்கோஸ் தயாரிக்கின்றன.",
        "କ୍ଲୋରୋଫିଲ୍ ସବୁଜ ଉଦ୍ଭିଦରେ ଆଲୋକ ସଂଶ୍ଳେଷଣ ପ୍ରକ୍ରିୟା ପାଇଁ ଏକ ଅତ୍ୟାବଶ୍ୟକ ଉପାଦାନ।",
        "जल चक्र पृथ्वी पर जल के निरंतर वाष्पीकरण, संघनन और वर्षण की प्रक्रिया है।",
        "মহাসাগরগুলি পৃথিবীর জলবায়ু নিয়ন্ত্রণে এবং কার্বন শোষণে গুরুত্বপূর্ণ ভূমিকা পালন করে।",
        "காடுகள் மண்ணரிப்பைத் தடுத்து பல்லுயிர் பெருக்கத்தைப் பாதுகாக்கின்றன.",
        "జీవవైవిధ్యం పర్యావరణ సమతుల్యతను కాపాడటంలో కీలక పాత్ర పోషిస్తుంది.",
        "अल्बर्ट आइंस्टीन ने सापेक्षता के सिद्धांत का प्रतिपादन किया था।",
        "ഐസക് ന്യൂട്ടൺ ഗുരുത്വാകർഷണ നിയമങ്ങളും ചലന നിയമങ്ങളും ആവിഷ്കരിച്ചു.",
        "ഗലീലിയോ ദൂരദർശിനി ഉപയോഗിച്ച് വ്യാഴത്തിന്റെ ഉപഗ്രഹങ്ങളെ കണ്ടെത്തി.",
        "মাদাম ক্যুরি তেজস্ক্রিয়তা নিয়ে গবেষণার জন্য দুটি ভিন্ন বিজ্ঞানের শাখায় নোবেল পেয়েছিলেন।",
        "సి. వి. రామన్ రామన్ ఎఫెక్ట్ ఆవిష్కరణకు గాను భౌతికశాస్త్రంలో నోబెల్ బహుమతి అందుకున్నారు.",
        "சுப்பிரமணிய பாரதியார் விடுதலைப் போராட்டக் காலத்தில் தீவிர தேசியவாதக் கவிதைகளை எழுதினார்.",
        "மிர்சா காಲಿப் உருது கவிதைகளில் தத்துவார்த்த சிந்தனைகளை வெளிப்படுத்திய ஒப்பற்ற கவிஞர்.",
        "मिर्ज़ा ग़ालिब उर्दू शायरी के बेमिसाल शायर थे जिनकी शायरी आज भी मशहूर है।",
        "मुंशी प्रेमचंद हिंदी साहित्य के महान उपन्यासकार और कहानीकार थे।",
        "গোদান উপন্যাসটি ভারতীয় গ্রামীণ সমাজের কৃষকদের দুঃখ-কষ্টের নিখুঁত চিত্রায়ন।",
        "ವಿಶ್ವೇಶ್ವರಯ್ಯನವರು ಭಾರತದ ಶ್ರೇಷ್ಠ ಇಂಜಿನಿಯರ್ ಮತ್ತು ಮುತ್ಸದ್ದಿ ಆಗಿದ್ದರು.",
        "மோக்ஷகுண்டம் விஸ்வேஸ்வரய்யாவின் பிறந்தநாள் இந்தியாவில் பொறியாளர் தினமாகக் கொண்டாடப்படுகிறது.",
        "మహాత్మా గాంధీ సత్యం మరియు అహింస ద్వారా భారత స్వాతంత్ర్య సమరాన్ని నడిపించారు.",
        "సుభాష్ చంద్రబోస్ 'నాకు రక్తం ఇవ్వండి, నేను మీకు స్వాతంత్ర్యం ఇస్తాను' అని నినదించారు.",
        "ഭഗത് സിംഗ് ചെറുപ്പത്തിലേ സ്വാതന്ത്ര്യ സമരത്തിൽ വീരമൃത്യു വരിച്ച വിപ്ലവകാരിയാണ്.",
        "ਸਰਦਾਰ ਵੱਲਭ ਭਾਈ ਪਟੇਲ ਨੇ ਭਾਰਤ ਦੀਆਂ ਸਾਰੀਆਂ ਰਿਆਸਤਾਂ ਨੂੰ ਇੱਕ ਸੁਤੰਤਰ ਗਣਰਾਜ ਵਿੱਚ ਜੋੜਿਆ।",
        "ਲਾਲ ਬਹਾਦਰ ਸ਼ਾਸਤਰੀ ਨੇ 'ਜੈ ਜਵਾਨ, ਜੈ ਕਿਸਾਨ' ਦਾ ਅਮਰ ਨਾਅਰਾ ਦਿੱਤਾ।",
        "అటల్ బిహారీ వాజ్‌పేయి భారత మాజీ ప్రధానమంత్రి మరియు గొప్ప వక్త.",
        "एपीजे अब्दुल कलाम भारत के मिसाइल मैन और जनता के राष्ट्रपति के रूप में प्रसिद्ध थे।",
        "వింగ్స్ ఆఫ్ ఫైర్ అబ్దుల్ కలాం గారి స్ఫూర్తిదాయక ఆత్మకథ.",
        "অগ্নিপক্ষ ড. কালামের আত্মজীবনী যুবসমাজের জন্য এক চিরন্তন অনুপ্রেরণা।",
        "নাসার জেমস ওয়েব টেলিস্কোপ দূরবর্তী মহাবিশ্বের নজিরবিহীন ছবি পাঠিয়েছে।",
        "ஹப்பிள் விண்வெளி தொலைநோக்கி பிரபஞ்சத்தின் விரிவாக்கத்தை அளவிட உதவியது.",
        "কৃষ্ণগহ্বর হলো মহাশূন্যের এমন এক স্থান যার মহাকর্ষ বল থেকে আলোও বের হতে পারে না।",
        "బ్లాక్ హోల్స్ గురుత్వాకర్షణ శక్తి అత్యంత శక్తివంతమైనది.",
        "न्यूट्रॉन तारे ब्रह्मांड में सबसे घने ज्ञात खगोलीय पिंडों में से हैं।",
        "পালসার হলো দ্রুত ঘূর্ণায়মান অত্যন্ত চৌম্বকীয় নিউট্রন তারা।",
        "సూపర్‌నోవా విస్ఫోటనం భారీ నక్షత్రాల అంతిమ దశలో సంభవిస్తుంది.",
        "सौर हवाएं पृथ्वी के चुंबकीय क्षेत्र से टकराकर ध्रुवीय ज्योति (ऑरोरा) बनाती हैं।",
        "சூரிய புயல்கள் பூமியின் தகவல் தொடர்பு அமைப்புகளை பாதிக்க வாய்ப்புள்ளது.",
        "ഭൗമകാന്തിക മണ്ഡലം ഭൂമിയെ മാരകമായ സൗരവികിരണങ്ങളിൽ നിന്ന് സംരക്ഷിക്കുന്നു.",
        "ஓசோன் படலம் சூரியனின் தீங்கு விளைவிக்கும் புற ஊதா கதிர்களை உறிஞ்சுகிறது.",
        "সিএফസി গ্যাসের অতিরিক্ত ব্যবহার ওজোন স্তর ক্ষয়ের প্রধান কারণ ছিল।",
        "আন্তর্জাতিক মহাকাশ স্টেশন (আইএসএস) পৃথিবীর কক্ষপথে স্থাপিত একটি স্থায়ী গবেষণাগার।",
        "స్పేస్ స్టేషన్ భూమి చుట్టూ ప్రతి 90 నిమిషాలకు ఒకసారి పరిభ్రమిస్తుంది.",
        "ಭೂಮಿಯ ವಾತಾವರಣದಲ್ಲಿ ಸಾರಜನಕ ಅನಿಲವು ಸುಮಾರು 78 ಪ್ರತಿಶತದಷ್ಟು ತುಂಬಿದೆ.",
        "ഓക്സിജൻ ജീവജാലങ്ങളുടെ ശ്വസനത്തിന് അത്യന്താപേക്ഷിതമായ വാതകമാണ്.",
        "கார்பன் டை ஆக்சைடு தாவரங்களின் ஒளிச்சேர்க்கைக்குத் தேவையான மூலப்பொருள்.",
        "హైడ్రోజన్ విశ్వంలో అత్యంత సమృద్ధిగా లభించే ప్రాథమిక మూలకం.",
        "হিলিয়াম পর্যায় সারণির দ্বিতীয় হালকা এবং একটি নিষ্ক্রিয় গ্যাস।",
        "लिथियम बैटरी आधुनिक इलेक्ट्रिक वाहनों और स्मार्टफोन्स की मुख्य ऊर्जा स्रोत हैं।",
        "સિલિકોન સેમિકન્ડક્ટર ઉદ્યોગમાં ચિપ ઉત્પાદન માટે સૌથી મહત્વનું તત્વ છે.",
        "ಗ್ರ್ಯಾಫೀನ್ ಅತ್ಯಂತ ಬಲಿಷ್ಠ, ತೆಳುವಾದ ಮತ್ತು ಶಾಖ ವಾಹಕ ವಸ್ತುವಾಗಿದೆ.",
        "కార్బన్ నానోట్యూబ్‌లు ఉక్కు కంటే వందల రెట్లు బలమైన నిర్మాణాలు.",
        "ദ്രാവക നൈട്രജൻ വളരെ താഴ്ന്ന താപനിലയിൽ വസ്തുക്കളെ തണുപ്പിക്കാൻ ഉപയോഗിക്കുന്നു.",
        "ਸੁਪਰ ਕੰਪਿਊਟਰ ਮੌਸਮ ਦੀ ਭਵਿੱਖਬਾਣੀ ਅਤੇ ਪਰਮਾਣੂ ਖੋਜ ਵਿੱਚ ਵਰਤੇ ਜਾਂਦੇ ਹਨ।",
        "ભારતનો પરમ સિદ્ધિ સુપર કોમ્પ્યુટર વૈશ્વિક સ્તરે ટોચના ક્રમમાં સ્થાન પામ્યો છે.",
        "ഇന്റർനെറ്റ് ഓഫ് തിങ്സ് (ഐഒടി) സ്മാർട്ട് ഉപകരണങ്ങളെ പരസ്പരം ബന്ധിപ്പിക്കുന്നു.",
        "5ஜி நெட்வொர்க் அதிவேக தரவு பரிமாற்றம் மற்றும் குறைந்த தாமதத்தை வழங்குகிறது.",
        "క్లౌడ్ కంప్యూటింగ్ ద్వారా డేటాను ఆన్‌లైన్ సర్వర్లలో సురక్షితంగా నిల్వ చేయవచ్చు.",
        "બ્લોકચેઇન ટેકનોલોજી સુરક્ષિત ડિજિટલ લેવડદેવડ અને ક્રિપ્ટોકરન્સી માટે આધાર છે.",
        "సైబర్ సెక్యూరిటీ డిజిటల్ డేటా మరియు నెట్‌వర్క్‌లను హ్యాకింగ్ నుండి కాపాడుతుంది.",
        "ડેટા સાયન્સ મોટા ડેટાસેટ્સમાંથી અર્થપૂર્ણ તારણો શોધવામાં મદદ કરે છે.",
        "റോബോട്ടിക്സ് വ്യാവസായിക ഉൽപാദനത്തിൽ കാര്യക്ഷമതയും കൃത്യതയും വർദ്ധിപ്പിക്കുന്നു.",
        "ಆಟೋಮೇಷನ್ ಕಾರ್ಖಾನೆಗಳಲ್ಲಿ ಮಾನವ ಶ್ರಮವನ್ನು ಕಡಿಮೆ ಮಾಡಿ ಉತ್ಪಾದನೆಯನ್ನು ಹೆಚ್ಚಿಸುತ್ತದೆ.",
        "ഡിജിറ്റൽ ഇന്ത്യ പദ്ധതി വഴി പൗരന്മാർക്ക് സർക്കാർ സേവനങ്ങൾ ഓൺലൈനായി ലഭ്യമാകുന്നു.",
        "ಯುಪಿಐ ಭಾರತದ ಡಿಜಿಟಲ್ ಪಾವತಿ ವ್ಯವಸ್ಥೆಯಲ್ಲಿ ಜಾಗತಿಕ ಕ್ರಾಂತಿಯನ್ನು ಸೃಷ್ಟಿಸಿದೆ.",
        "ಭೀಮ್ ಆಪ್ ಮತ್ತು ಯುಪಿಐ ಮೂಲಕ ಸೆಕೆಂಡುಗಳಲ್ಲಿ ಹಣ ವರ್ಗಾವಣೆ ಸಾಧ್ಯವಾಗಿದೆ.",
        "જન ધન યોજના દ્વારા કરોડો નાગરિકોને બેંકિંગ સેવાઓ સાથે જોડવામાં આવ્યા.",
        "આયુષ્માન ભારત યોજના ગરીબ પરિવારોને મફત સ્વાસ્થ્ય વીમો પૂરો પાડે છે.",
        "స్వచ్ఛ భారత్ అభియాన్ దేశవ్యాప్తంగా పరిశుభ్రతపై ప్రజల్లో చైతన్యం తెచ్చింది.",
        "നമാമി ഗംഗേ പദ്ധതി ഗംഗാ നദിയുടെ ശുദ്ധീകരണത്തിനും സംരക്ഷണത്തിനുമായി ആരംഭിച്ചു.",
        "ಪ್ರಧಾನ ಮಂತ್ರಿ ಆವಾಸ್ ಯೋಜನೆ ಪ್ರತಿಯೊಬ್ಬರಿಗೂ ಶಾಶ್ವತ ಮನೆ ಒದಗಿಸುವ ಗುರಿ ಹೊಂದಿದೆ.",
        "ಉಜ್ವಲ ಯೋಜನೆಯು ಗ್ರಾಮೀಣ ಮಹಿಳೆಯರಿಗೆ ಉಚಿತ ಎಲ್ಪಿಜಿ ಸಿಲಿಂಡರ್ ಒದಗಿಸಿದೆ.",
        "ಕಿಸಾನ್ ಸಮ್ಮಾನ್ ನಿಧಿ ರೈತರಿಗೆ ಆರ್ಥಿಕ ಬೆಂಬಲ ನೀಡುವ ಮಹತ್ವದ ಯೋಜನೆಯಾಗಿದೆ.",
        "സ്റ്റാർട്ടപ്പ് ഇന്ത്യ സംരംഭം രാജ്യത്തെ യുവാക്കളിൽ സംരംഭകത്വം പ്രോത്സാഹിപ്പിക്കുന്നു.",
        "മേക്ക് ഇൻ ഇന്ത്യ പദ്ധതി രാജ്യത്തെ ആഗോള ഉൽപാദന കേന്ദ്രമാക്കാൻ ലക്ഷ്യമിടുന്നു.",
        "ആത്മനിർഭർ ഭാരത് വിവിധ മേഖലകളിൽ ഇന്ത്യയെ സ്വയംപര്യാപ്തമാക്കാൻ സഹായിക്കുന്നു.",
        "வளர்ச்சி அடைந்த பாரதம் 2047 என்ற தொலைநோக்குப் பார்வை நாட்டை முன்னேற்றப் பாதையில் கொண்டு செல்கிறது.",
        "భారతదేశ జిడిపి ప్రపంచంలోనే అత్యంత వేగంగా వృద్ధి చెందుతున్న ప్రధాన ఆర్థిక వ్యవస్థలలో ఒకటి.",
        "రిజర్వ్ బ్యాంక్ ఆఫ్ ఇండియా దేశ ద్రవ్య విధానాన్ని మరియు బ్యాంకింగ్ వ్యవస్థను నియంత్రిస్తుంది.",
        "స్టాక్ మార్కెట్లలో సెబీ (SEBI) పెట్టుబడిదారుల ప్రయోజనాలను రక్షిస్తుంది.",
        "જીએસટી (GST) એક રાષ્ટ્ર એક કર પ્રણાલી તરીકે ભારતમાં લાગુ કરવામાં આવી.",
        "પ્રત્યક્ષ કર સંગ્રહમાં ભારતમાં સતત નોંધપાત્ર વૃદ્ધિ જોવા મળી રહી છે.",
        "ਭਾਰਤੀ ਬਰਾਮਦਾਂ ਵਿੱਚ ਦਵਾਈਆਂ, ਸਾਫਟਵੇਅਰ ਅਤੇ ਇੰਜੀਨੀਅਰਿੰਗ ਉਤਪਾਦ ਮੋਹਰੀ ਹਨ।",
        "ਸੂਖਮ, ਲਘੂ ਅਤੇ ਦਰਮਿਆਨੇ ਉਦਯੋਗ (MSME) ਰੁਜ਼ਗਾਰ ਸਿਰਜਣ ਵਿੱਚ ਵੱਡਾ ਯੋਗਦਾਨ ਪਾਉਂਦੇ ਹਨ।",
        "ਭਾਰਤ ਵਿਸ਼ਵ ਦਾ ਸਭ ਤੋਂ ਵੱਡਾ ਜੈਨੇਰਿਕ ਦਵਾਈਆਂ ਦਾ ਉਤਪਾਦਕ ਦੇਸ਼ ਹੈ।",
        "ਕੋਵੈਕਸੀਨ ਅਤੇ ਕੋਵੀਸ਼ੀਲਡ ਨੇ ਕੋਰੋਨਾ ਮਹਾਂਮਾਰੀ ਦੌਰਾਨ ਲੱਖਾਂ ਜਾਨਾਂ ਬਚਾਈਆਂ।",
        "ਪੋਲੀਓ ਮੁਕਤ ਭਾਰਤ ਅਭਿਆਨ ਵਿਸ਼ਵ ਸਿਹਤ ਸੰਗਠਨ ਦੁਆਰਾ ਸ਼ਲਾਘਾਯੋਗ ਸਫਲਤਾ ਮੰਨਿਆ ਗਿਆ।",
        "ਮਲੇਰੀਆ ਅਤੇ ਟੀਬੀ ਦੇ ਖਾਤਮੇ ਲਈ ਭਾਰਤ ਸਰਕਾਰ ਲਗਾਤਾਰ ਠੋਸ ਕਦਮ ਚੁੱਕ ਰਹੀ ਹੈ।",
        "ਮਾਨਸਿਕ ਸਿਹਤ ਵੀ ਸਰੀਰਕ ਸਿਹਤ ਵਾਂਗ ਹੀ ਜੀਵਨ ਵਿੱਚ ਬਹੁਤ ਜ਼ਰੂਰੀ ਹੈ।",
        "ਯੋਗਾ ਅਤੇ ਪ੍ਰਾਣਾਯਾਮ ਮਾਨਸਿਕ ਤਣਾਅ ਨੂੰ ਦੂਰ ਕਰਕੇ ਇਕਾਗਰਤਾ ਵਧਾਉਂਦੇ ਹਨ।",
        "ਅੰਤਰਰਾਸ਼ਟਰੀ ਯੋਗ ਦਿਵਸ ਹਰ ਸਾਲ 21 ਜੂਨ ਨੂੰ ਵਿਸ਼ਵ ਭਰ ਵਿੱਚ ਮਨਾਇਆ ਜਾਂਦਾ ਹੈ।",
        "ਸੰਸਕ੍ਰਿਤ ਭਾਰਤ ਦੀ ਇੱਕ ਪ੍ਰਾਚੀਨ ਅਤੇ ਬਹੁਤ ਵਿਗਿਆਨਕ ਭਾਸ਼ਾ ਮੰਨੀ ਜਾਂਦੀ ਹੈ।",
        "ਵੇਦ, ਉਪਨਿਸ਼ਦ ਅਤੇ ਪੁਰਾਣ ਭਾਰਤੀ ਦਰਸ਼ਨ ਦੇ ਅਮੀਰ ਖ਼ਜ਼ਾਨੇ ਹਨ।",
        "ਮਹਾਭਾਰਤ ਅਤੇ ਰਾਮਾਇਣ ਭਾਰਤ ਦੇ ਦੋ ਮਹਾਨ ਪ੍ਰਾਚੀਨ ਮਹਾਕਾਵਿ ਹਨ।",
        "ਭਗਵਦ ਗੀਤਾ ਨਿਸ਼ਕਾਮ ਕਰਮ ਅਤੇ ਧਰਮ ਦੇ ਮਾਰਗ 'ਤੇ ਚੱਲਣ ਦੀ ਸਿੱਖਿਆ ਦਿੰਦੀ ਹੈ।",
        "ਬੁੱਧ ਧਰਮ ਨੇ ਅਹਿੰਸਾ, ਕਰੁਣਾ ਅਤੇ ਅਸ਼ਟਾਂਗ ਮਾਰਗ ਦਾ ਸੰਦੇਸ਼ ਪੂਰੀ ਦੁਨੀਆ ਵਿੱਚ ਫੈਲਾਇਆ।",
        "ਜੈਨ ਧਰਮ ਵਿੱਚ ਅਹਿੰਸਾ ਪਰਮੋ ਧਰਮ ਨੂੰ ਸਭ ਤੋਂ ਉੱਚਾ ਸਿਧਾਂਤ ਮੰਨਿਆ ਗਿਆ ਹੈ।",
        "ਭਗਵਾਨ ਮਹਾਵੀਰ ਨੇ ਤ੍ਰਿਰਤਨ - ਸਮਯਕ ਦਰਸ਼ਨ, ਸਮਯਕ ਗਿਆਨ ਅਤੇ ਸਮਯਕ ਚਰਿੱਤਰ ਦਾ ਉਪਦੇਸ਼ ਦਿੱਤਾ।",
        "ਸਿੱਖ ਧਰਮ ਦੀ ਸਥਾਪਨਾ ਸ੍ਰੀ ਗੁਰੂ ਨਾਨਕ ਦੇਵ ਜੀ ਨੇ ਸਰਬੱਤ ਦੇ ਭਲੇ ਲਈ ਕੀਤੀ ਸੀ।",
        "ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ ਜੀ ਵਿੱਚ ਵੱਖ-ਵੱਖ ਭਗਤਾਂ ਅਤੇ ਗੁਰੂ ਸਾਹਿਬਾਨ ਦੀ ਪਵਿੱਤਰ ਬਾਣੀ ਦਰਜ ਹੈ।",
        "ਲੰਗਰ ਪ੍ਰਥਾ ਸਮਾਜ ਵਿੱਚ ਬਰਾਬਰੀ ਅਤੇ ਭਾਈਚਾਰੇ ਦਾ ਅਦਭੁਤ ਪ੍ਰਤੀਕ ਹੈ।",
        "ਸੂਫ਼ੀ ਸੰਤਾਂ ਨੇ ਪਿਆਰ, ਸਹਿਣਸ਼ੀਲਤਾ ਅਤੇ ਭਾਈਚਾਰੇ ਦਾ ਸੁਨੇਹਾ ਲੋਕਾਂ ਤੱਕ ਪਹੁੰਚਾਇਆ।",
        "ਅਮੀਰ ਖੁਸਰੋ ਨੇ ਹਿੰਦਵੀ ਅਤੇ ਕੱਵਾਲੀ ਸੰਗੀਤ ਨੂੰ ਇੱਕ ਨਵੀਂ ਪਛਾਣ ਦਿੱਤੀ।",
    ]

    queries: List[str] = [pair[0] for pair in seed_parallel_pairs]
    corpus_docs: List[str] = [pair[1] for pair in seed_parallel_pairs] + distractor_documents
    language_labels: List[str] = [pair[2] for pair in seed_parallel_pairs]

    return queries, corpus_docs, language_labels


def benchmark_uncompressed_search(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    num_warmup: int = 5,
) -> Tuple[torch.Tensor, float]:
    """Compute exact float32 ground-truth similarities and measure per-query search latency.

    Args:
        query_embeddings: Normalized queries of shape [Q, d].
        corpus_embeddings: Normalized corpus vectors of shape [N, d].
        num_warmup: Number of warmup queries.

    Returns:
        Tuple of (ground_truth_top1_indices [Q], mean_latency_ms).
    """
    num_queries = query_embeddings.shape[0]

    # Warmup
    for i in range(min(num_warmup, num_queries)):
        q = query_embeddings[i : i + 1]
        _ = torch.matmul(q, corpus_embeddings.T)
        _ = torch.topk(_, k=1, dim=-1)

    # Measure exact per-query uncompressed search latency
    latencies: List[float] = []
    top1_indices_list: List[int] = []

    for i in range(num_queries):
        q = query_embeddings[i : i + 1]
        t0 = time.perf_counter()
        sims = torch.matmul(q, corpus_embeddings.T)
        top1 = torch.topk(sims, k=1, dim=-1).indices
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)
        top1_indices_list.append(int(top1[0, 0].item()))

    mean_latency_ms = sum(latencies) / len(latencies)
    ground_truth_top1 = torch.tensor(top1_indices_list, dtype=torch.long)

    return ground_truth_top1, mean_latency_ms


def evaluate_turboquant_configuration(
    corpus_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    ground_truth_top1: torch.Tensor,
    k_candidates: int,
    use_qjl: bool,
    num_warmup: int = 5,
) -> Dict[str, Any]:
    """Evaluate TurboQuantIndex for a specific (k_candidates, use_qjl) configuration.

    Tracks:
        - Recall@K: Proportion of queries where true float32 top-1 document is in Stage 1's top-K.
        - Top-1 Agreement: Proportion of queries where re-ranked top document matches true top-1.
        - Query Latency (ms): Mean search time per query.

    Returns:
        Dict with metrics and timing.
    """
    d = corpus_embeddings.shape[-1]
    num_queries = query_embeddings.shape[0]

    # Initialize index
    index = TurboQuantIndex(d=d, k_candidates=k_candidates, use_qjl=use_qjl)
    index.add(corpus_embeddings)

    # Warmup
    for i in range(min(num_warmup, num_queries)):
        q = query_embeddings[i : i + 1]
        _ = index.search(q, top_k=1)

    stage1_hits = 0
    top1_matches = 0
    search_latencies: List[float] = []

    for i in range(num_queries):
        q = query_embeddings[i : i + 1]
        true_top1_idx = ground_truth_top1[i].item()

        # 1. Stage 1 Coarse Filter: candidate pool retrieval
        _, candidate_indices = index.coarse_filter(q, k=k_candidates)
        candidate_ids = candidate_indices[0].tolist()
        if true_top1_idx in candidate_ids:
            stage1_hits += 1

        # 2. Stage 2 Two-stage search end-to-end (timed)
        t0 = time.perf_counter()
        search_res = index.search(q, top_k=1)
        t1 = time.perf_counter()
        search_latencies.append((t1 - t0) * 1000.0)

        retrieved_top1 = search_res.indices[0, 0].item()
        if retrieved_top1 == true_top1_idx:
            top1_matches += 1

    recall_at_k = (stage1_hits / num_queries) * 100.0
    top1_agreement = (top1_matches / num_queries) * 100.0
    mean_latency_ms = sum(search_latencies) / len(search_latencies)

    return {
        "k_candidates": k_candidates,
        "use_qjl": use_qjl,
        "recall_at_k": recall_at_k,
        "top1_agreement": top1_agreement,
        "mean_latency_ms": mean_latency_ms,
    }


def format_markdown_table(
    results: List[Dict[str, Any]],
    baseline_latency_ms: float,
) -> str:
    """Format evaluation benchmark results into a clean markdown table."""
    headers = [
        "Method",
        "QJL Correction",
        "Stage 1 Pool (K)",
        "Recall@K (%)",
        "Top-1 Agreement (%)",
        "Mean Latency (ms)",
        "Relative to FP32",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        f"| Full-Precision Float32 | N/A | Full Corpus | 100.00% | 100.00% | {baseline_latency_ms:.4f} ms | 1.00x (Baseline) |",
    ]

    for res in results:
        qjl_str = "Enabled (True)" if res["use_qjl"] else "Disabled (False)"
        k_val = res["k_candidates"]
        recall = f"{res['recall_at_k']:.2f}%"
        agreement = f"{res['top1_agreement']:.2f}%"
        lat = f"{res['mean_latency_ms']:.4f} ms"
        speedup = (
            f"{baseline_latency_ms / res['mean_latency_ms']:.2f}x"
            if res["mean_latency_ms"] > 0
            else "N/A"
        )

        row = [
            "TurboQuantIndex",
            qjl_str,
            str(k_val),
            recall,
            agreement,
            lat,
            speedup,
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def run_retrieval_benchmark() -> None:
    """Run comprehensive retrieval evaluation benchmark and output report."""
    print("=" * 80)
    print(" TurboQuant vs. Full-Precision Float32 Retrieval Benchmark on Indic Text")
    print("=" * 80)

    # 1. Corpus & Queries setup
    print("\n[1/5] Loading Indic Multilingual Evaluation Dataset...")
    queries, corpus_docs, lang_labels = build_indic_evaluation_dataset()
    unique_langs = sorted(list(set(lang_labels)))
    print(f"      Corpus size : {len(corpus_docs)} documents")
    print(f"      Query set   : {len(queries)} queries")
    print(f"      Languages   : {', '.join(unique_langs)}")

    # 2. Embedding Extraction
    print("\n[2/5] Initializing IndicBERT Embedding Extractor...")
    model, tokenizer, model_info = load_indic_bert_model("ai4bharat/indic-bert")
    print(f"      Embedding Model : {model_info}")

    print("      Extracting and L2-normalizing corpus embeddings...")
    corpus_embs = extract_indic_embeddings(corpus_docs, model, tokenizer)
    print(f"      Corpus embeddings shape: {tuple(corpus_embs.shape)} (dtype={corpus_embs.dtype})")

    print("      Extracting and L2-normalizing query embeddings...")
    query_embs = extract_indic_embeddings(queries, model, tokenizer)
    print(f"      Query embeddings shape : {tuple(query_embs.shape)} (dtype={query_embs.dtype})")

    # Verify L2 normalization
    c_norms = torch.norm(corpus_embs, p=2, dim=-1)
    q_norms = torch.norm(query_embs, p=2, dim=-1)
    assert torch.allclose(c_norms, torch.ones_like(c_norms), atol=1e-4)
    assert torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-4)
    print("      L2 normalization verified: all vectors have unit length ||v||_2 = 1.0.")

    # 3. Ground Truth Baseline
    print("\n[3/5] Computing Full-Precision Float32 Exact Dot-Product Baseline...")
    ground_truth_top1, fp32_latency_ms = benchmark_uncompressed_search(query_embs, corpus_embs)
    print(f"      Float32 Baseline Mean Latency: {fp32_latency_ms:.4f} ms/query")

    # 4. Experimental Grid
    candidate_pools = [10, 25, 50, 100]
    qjl_settings = [False, True]
    total_experiments = len(candidate_pools) * len(qjl_settings)

    print(f"\n[4/5] Executing Experimental Grid ({total_experiments} configurations)...")
    print("      Candidate pool sizes K : [10, 25, 50, 100]")
    print("      Quantization modes     : Standard 2-bit (QJL=False) vs QJL-Corrected (QJL=True)")

    results: List[Dict[str, Any]] = []

    exp_idx = 1
    for use_qjl in qjl_settings:
        mode_desc = "QJL-Corrected (use_qjl=True)" if use_qjl else "Standard 2-bit (use_qjl=False)"
        print(f"\n  --- Mode: {mode_desc} ---")
        for k in candidate_pools:
            res = evaluate_turboquant_configuration(
                corpus_embeddings=corpus_embs,
                query_embeddings=query_embs,
                ground_truth_top1=ground_truth_top1,
                k_candidates=k,
                use_qjl=use_qjl,
            )
            results.append(res)
            print(
                f"    [{exp_idx:02d}/{total_experiments}] K={k:3d} | "
                f"Recall@K: {res['recall_at_k']:6.2f}% | "
                f"Top-1 Agreement: {res['top1_agreement']:6.2f}% | "
                f"Latency: {res['mean_latency_ms']:.4f} ms"
            )
            exp_idx += 1

    # 5. Formatted Markdown Report
    print("\n[5/5] Benchmark Report Summary:\n")
    markdown_table = format_markdown_table(results, fp32_latency_ms)
    print(markdown_table)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_retrieval_benchmark()
