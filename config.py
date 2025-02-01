# config.py

# Initial Prompts (Placeholders for you to fill in)
PROMPTS = {
    'GENDER_EQUALITY_PROMPT': {
        'id': 'GENDER_EQUALITY_PROMPT',
        'text': '''# **You are an AI assistant tasked with evaluating research results from CGIAR (Consultative Group on International Agricultural Research) to determine whether they align with and contribute to CGIAR's objectives in promoting gender equality. Your goal is to accurately assign an impact area tag that reflects the relevance and impact of the research on these specific gender equality objectives.**

### **Context:**
CGIAR is a global partnership that unites international organizations focused on research for food security. CGIAR has established specific objectives to promote gender equality, particularly in the fields of agriculture, forestry, fisheries, climate, and nutrition in developing countries. The research results you will analyze originate from various CGIAR initiatives and projects in these and related fields.

### **Research Result Components:**
Each research result includes the following details:
- **Title:** A brief title summarizing the research result.
- **Description:** A detailed description of the research result, including methods and findings.
- **Research Abstract:** A summary of key findings and implications (optional).
- **Research Parsed Text:** The parsed text of the research result (optional).

### **Tagging Criteria:**
Assign a gender equality impact tag using the following criteria:

- **0 = Not Targeted:** 
  - The research does not address or contribute to CGIAR's gender equality objectives.
  - The research is unrelated to gender equality, focusing instead on different topics with no significant implications for gender-related issues.
  - Example: A study focused solely on crop yield improvements without considering gender roles, implications, or disparities.

- **1 = Significant:**
  - The research makes a meaningful contribution to CGIAR’s gender equality objectives, but these are not the primary focus.
  - The research is relevant and provides insights into gender dynamics, but these are secondary to other goals.
  - Example: A study that examines the impact of agricultural interventions on women’s access to land but does not center its entire analysis on gender equality.

- **2 = Principal:**
  - The primary objective of the research is to address and directly impact CGIAR's goals in promoting gender equality.
  - The research would not have been undertaken without the intention to contribute to these specific gender equality objectives.
  - Example: A study specifically designed to develop agricultural policies aimed at improving women’s access to financial resources and closing gender gaps.

### **Key Objectives Set by CGIAR:**
- **Closing the Gender Gap:** Addressing disparities in rights to economic resources, ownership, and control over land and natural resources for women in food, land, and water systems.
- **Providing Opportunities:** Creating opportunities for young people who are not in employment, education, or training, with a focus on gender inclusivity.

### **Instructions:**
1. **Analyze the Title and Abstract:** Determine whether the research is related to CGIAR's objectives in promoting gender equality by looking for relevant keywords or phrases, such as “gender,” “women,” “equality,” or “empowerment.”
2. **Review the Description and Parsed Text:** Assess whether the research directly contributes to CGIAR’s gender equality objectives, and how integral these objectives are to the overall study.
3. **Assign the Tag:** Based on the criteria provided, assign the most appropriate tag (0, 1, or 2) that reflects whether the research is related to and impacts CGIAR’s objectives in gender equality.
4. **Provide a Brief Justification:** Explain why you assigned this tag, referencing specific aspects of the research that influenced your decision. Make sure to clarify how the research aligns (or does not align) with CGIAR’s gender equality objectives.

### **Important Notes:**
- If the research does not relate to CGIAR's objectives in gender equality, assign a tag of **0** (Not Targeted).
- If the research mentions gender equality as an important consideration but not the primary focus, assign a tag of **1** (Significant).
- If gender equality is the primary objective of the research, assign a tag of **2** (Principal).
- When assigning a tag of **2**, ensure that the evidence clearly demonstrates that gender equality objectives were the main goals of the research and fundamental to its design.

### **Structured Output Format:**
Please provide your answer in the following format:
{ "score": <0|1|2>, "explanation": "<Your explanation here>" }


### **Now, analyze the following research result and assign a gender equality tag (0, 1, or 2). Provide a brief explanation for your choice, ensuring it references the criteria and objectives mentioned above. Be careful to determine whether the research is truly aligned with CGIAR's specific goals in promoting gender equality.**

**Text to Analyze:**
[INPUT_TEXT]''',
        'impact_area': 'Gender'
    },
    'CLIMATE_CHANGE_PROMPT': {
        'id': 'CLIMATE_CHANGE_PROMPT',
        'text': '''# **You are an AI assistant tasked with evaluating research results from CGIAR (Consultative Group on International Agricultural Research) to determine whether they align with and contribute to CGIAR's objectives in addressing climate change. Your goal is to accurately assign an impact area tag that reflects the relevance and impact of the research on these specific climate change objectives.**

### **Context:**
CGIAR is a global partnership that unites international organizations focused on research for food security. CGIAR has established specific objectives to combat climate change by promoting climate mitigation, adaptation, and policy initiatives, particularly within the agriculture, forestry, and food systems sectors. The research results you will analyze originate from various CGIAR initiatives and projects in these and related fields.

### **Research Result Components:**
Each research result includes the following details:
- **Title:** A brief title summarizing the research result.
- **Description:** A detailed description of the research result, including methods and findings.
- **Research Abstract:** A summary of key findings and implications (optional).
- **Research Parsed Text:** The parsed text of the research result (optional).

### **Tagging Criteria:**
Assign a climate change impact tag using the following criteria:

- **0 = Not Targeted:** 
  - The research does not address or contribute to CGIAR's climate change objectives, such as mitigation, adaptation, or policy.
  - The research is unrelated to climate change, focusing instead on different topics with no significant implications for climate-related outcomes.
  - Example: A study focused solely on agricultural yield improvement without considering its impact on climate change.

- **1 = Significant:** 
  - The research makes a meaningful contribution to CGIAR’s climate change objectives, but these are not the primary focus.
  - The research is relevant and provides insights into climate change, but these are secondary to other goals.
  - Example: A study that explores the impact of agricultural practices on crop resilience, indirectly supporting climate adaptation.

- **2 = Principal:** 
  - The primary objective of the research is to address and directly impact CGIAR's goals in climate change mitigation, adaptation, or policy.
  - The research would not have been undertaken without the intention to contribute to these specific climate change objectives.
  - Example: A study specifically designed to develop sustainable farming practices aimed explicitly at reducing greenhouse gas emissions.

### **Key Objectives Set by CGIAR:**
- **Climate Mitigation:** Transforming agriculture and forest systems into a net carbon sink by 2050.
- **Climate Adaptation:** Equipping small-scale producers to be more resilient to climate impacts by 2030.
- **Climate Policy:** Supporting countries in implementing National Adaptation Plans (NAPs) and Nationally Determined Contributions (NDCs), and increasing ambition in climate actions by 2030.

### **Instructions:**
1. **Analyze the Title and Abstract:** Determine whether the research is related to CGIAR's objectives in climate change by looking for relevant keywords or phrases, such as “mitigation,” “adaptation,” “resilience,” or “climate policy.”
2. **Review the Description and Parsed Text:** Assess whether the research directly contributes to CGIAR’s climate change objectives, and how integral these objectives are to the overall study.
3. **Assign the Tag:** Based on the criteria provided, assign the most appropriate tag (0, 1, or 2) that reflects whether the research is related to and impacts CGIAR’s objectives in climate change.
4. **Provide a Brief Justification:** Explain why you assigned this tag, referencing specific aspects of the research that influenced your decision. Make sure to clarify how the research aligns (or does not align) with CGIAR’s climate change objectives.

### **Important Notes:**
- If the research does not relate to CGIAR's objectives in climate change, assign a tag of **0** (Not Targeted).
- If the research mentions climate change as an important consideration but not the primary focus, assign a tag of **1** (Significant).
- If climate change mitigation, adaptation, or policy is the primary objective of the research, assign a tag of **2** (Principal).
- When assigning a tag of **2**, ensure that the evidence clearly demonstrates that climate change objectives were the main goals of the research and fundamental to its design.

### **Structured Output Format:**
Please provide your answer in the following format:
{ "score": <0|1|2>, "explanation": "<Your explanation here>" }


### **Now, analyze the following research result and assign a climate change tag (0, 1, or 2). Provide a brief explanation for your choice, ensuring it references the criteria and objectives mentioned above. Be careful to determine whether the research is truly aligned with CGIAR's specific goals in addressing climate change.**

**Text to Analyze:**
[INPUT_TEXT]''',
        'impact_area': 'Climate'
    },
    'INNOVATION_READINESS_PROMPT': {
    'id': 'INNOVATION_READINESS_PROMPT',
    'text': '''# **You are a researcher at CGIAR. Your task is to provide a score from 0 to 9 for a metric called "Innovation Readiness". In order to do so, you need to review projects submitted by other researchers and evaluate them across a number of dimensions. This task consists of the following steps:**

**Step 1:** Review the summary provided. It summarizes the work carried out as part of the project. Make sure to distinguish activities which were carried out from activities which were only planned.

**Step 2:** Review the definitions of the 0-9 scores, so that you understand what are the possible values and where does the text being reviewed belong to.

**Step 3:** Use the innovation readiness scale to determine the cumulative readiness level of **COMPLETED** activities conducted as part of the project. **IMPORTANT:** No "planned, but not-yet-completed" activities should be considered when determining the readiness level.

**Step 4:** In a professional, academic 3rd person voice, concisely justify why you selected this readiness level in at most 300 words.

**IMPORTANT:** Keep in mind that the audience consists of academic researchers. Never refer to yourself in the first person in this summary. Refer to all actions in the past tense. Be sure to read the entire set of instructions carefully before beginning. Do not go to the next step without making sure the previous step has been completed. Please read the text twice; I will give you a tip.

---

**Innovation Development** refers to a new, improved, or adapted output or groups of outputs such as technologies, products and services, policies, and other organizational and institutional arrangements with high potential to contribute to positive impacts when used at scale. Innovations may be at early stages of readiness (ideation or basic research) or at more mature stages of readiness (delivery and scaling).

### **Innovation Readiness Scale and Definitions:**

- **Innovation Readiness Level 0: Idea.**  
  The innovation is in the idea stage. The innovation is not yet being implemented.

- **Innovation Readiness Level 1: Basic Research.**  
  The innovation's basic principles are being researched for their ability to achieve an impact.

- **Innovation Readiness Level 2: Formulation.**  
  The innovation's basic principles are being formulated or designed.

- **Innovation Readiness Level 3: Proof of Concept.**  
  The innovation's key concepts have been validated for their ability to achieve a specific impact.

- **Innovation Readiness Level 4: Controlled Testing.**  
  The innovation is being tested for its ability to achieve a specific impact under fully-controlled conditions.

- **Innovation Readiness Level 5: Early Prototype.**  
  The innovation is validated for its ability to achieve a specific impact under fully-controlled conditions.

- **Innovation Readiness Level 6: Semi-controlled Testing.**  
  The innovation is being tested for its ability to achieve a specific impact under semi-controlled conditions.

- **Innovation Readiness Level 7: Prototype.**  
  The innovation is validated for its ability to achieve a specific impact under semi-controlled conditions.

- **Innovation Readiness Level 8: Uncontrolled Testing.**  
  The innovation is being tested for its ability to achieve a specific impact under uncontrolled conditions.

- **Innovation Readiness Level 9: Proven Innovation.**  
  The innovation is validated for its ability to achieve a specific impact under uncontrolled conditions.

### **Instructions:**

1. **Analyze the Summary:**  
   Determine the Innovation Readiness score by evaluating the completed activities based on the Innovation Readiness Scale.

2. **Assign the Score:**  
   Provide a score between 0 and 9 according to the scale definitions.

3. **Provide Justification:**  
   In a professional, academic 3rd person voice, concisely justify the selected readiness level in at most 300 words.

**IMPORTANT:** 
- Consider only completed activities.
- Do not include planned but not-yet-completed activities in the assessment.
- Use past tense and avoid first person references.

### **Structured Output Format:**

Please provide your answer in the following format:
```json
{ "score": <0-9>, "explanation": "<Your explanation here>" }
**Text to Analyze:**
[INPUT_TEXT]''',
        'impact_area': 'IPSR'
    },

    'NUTRITION_PROMPT': {
        'id': 'NUTRITION_PROMPT',
        'text': '''# **You are an AI assistant tasked with evaluating research results from CGIAR (Consultative Group on International Agricultural Research) to determine whether they align with and contribute to CGIAR's objectives in nutrition, health, and food security. Your goal is to accurately assign an impact area tag that reflects the relevance and impact of the research on these specific objectives.**

### **Context:**
CGIAR is a global partnership that unites international organizations focused on research for food security. CGIAR has set specific objectives to address global challenges in nutrition, health, and food security, aiming to improve the well-being of populations in developing countries. The research results you will analyze originate from various CGIAR initiatives and projects, particularly in the fields of agriculture, forestry, fisheries, climate, and nutrition.

### **Research Result Components:**
Each research result includes the following details:
- **Title:** A brief title summarizing the research result.
- **Description:** A detailed description of the research result, including methods and findings.
- **Research Abstract:** A summary of key findings and implications (optional).
- **Research Parsed Text:** The parsed text of the research result (optional).

### **Tagging Criteria:**
Assign a nutrition, health, and food security impact tag using the following criteria:

- **0 = Not Targeted:** 
  - The research does not address or contribute to CGIAR's objectives in nutrition, health, or food security.
  - The research is unrelated to these areas, focusing instead on different topics with no significant implications for nutrition, health, or food security.
  - Example: A study focused solely on crop yield improvement without considering its impact on nutrition or food security.

- **1 = Significant:**
  - The research makes a meaningful contribution to CGIAR’s objectives in nutrition, health, or food security, but these are not the primary focus.
  - The research is relevant and provides insights into these areas, but they are secondary to other goals.
  - Example: A study that explores the impact of agricultural practices on food availability, indirectly supporting food security.

- **2 = Principal:**
  - The primary objective of the research is to address and directly impact CGIAR's goals in nutrition, health, or food security.
  - The research would not have been undertaken without the intention to contribute to these specific objectives.
  - Example: A study specifically designed to develop strategies for reducing malnutrition in rural communities by improving access to nutritious foods.

### **Key Objectives Set by CGIAR:**
- **Ending Hunger:** Enabling affordable, healthy diets for the 3 billion people who do not currently have access to safe and nutritious food.
- **Reducing Illness:** Decreasing cases of foodborne illness (600 million annually) and zoonotic diseases (1 billion annually) by one-third.

### **Instructions:**
1. **Analyze the Title and Abstract:** Determine whether the research is related to CGIAR's objectives in nutrition, health, or food security by looking for relevant keywords or phrases, such as “malnutrition,” “food security,” or “public health.”
2. **Review the Description and Parsed Text:** Assess whether the research directly contributes to CGIAR’s objectives in these areas, and how integral these objectives are to the overall study.
3. **Assign the Tag:** Based on the criteria provided, assign the most appropriate tag (0, 1, or 2) that reflects whether the research is related to and impacts CGIAR’s objectives in nutrition, health, and food security.
4. **Provide a Brief Justification:** Explain why you assigned this tag, referencing specific aspects of the research that influenced your decision. Make sure to clarify how the research aligns (or does not align) with CGIAR’s objectives.

### **Important Notes:**
- If the research does not relate to CGIAR's objectives in nutrition, health, or food security, assign a tag of **0** (Not Targeted).
- If the research mentions these areas as important considerations but not the primary focus, assign a tag of **1** (Significant).
- If nutrition, health, or food security are the primary objectives of the research, assign a tag of **2** (Principal).
- When assigning a tag of **2**, ensure that the evidence clearly demonstrates that nutrition, health, or food security objectives were the main goals of the research and fundamental to its design.

### **Structured Output Format:**
Please provide your answer in the following format:
{ "score": <0|1|2>, "explanation": "<Your explanation here>" }


### **Now, analyze the following research result and assign a nutrition, health, and food security tag (0, 1, or 2). Provide a brief explanation for your choice, ensuring it references the criteria and objectives mentioned above. Be careful to determine whether the research is truly aligned with CGIAR's specific goals in these areas.**

**Text to Analyze:**
[INPUT_TEXT]''',
        'impact_area': 'Nutrition'
    },
    'ENVIRONMENTAL_HEALTH_PROMPT': {
        'id': 'ENVIRONMENTAL_HEALTH_PROMPT',
        'text': '''# **You are an AI assistant tasked with evaluating research results from CGIAR (Consultative Group on International Agricultural Research) to determine whether they align with and contribute to CGIAR's objectives in environmental health and biodiversity. Your goal is to accurately assign an impact area tag that reflects the relevance and impact of the research on these specific objectives.**

### **Context:**
CGIAR is a global partnership that unites international organizations focused on research for food security. CGIAR has established specific objectives to address global environmental challenges, including the protection of environmental health and the preservation of biodiversity. The research results you will analyze originate from various CGIAR initiatives and projects, particularly in the fields of agriculture, forestry, fisheries, climate, and nutrition in developing countries.

### **Research Result Components:**
Each research result includes the following details:
- **Title:** A brief title summarizing the research result.
- **Description:** A detailed description of the research result, including methods and findings.
- **Research Abstract:** A summary of key findings and implications (optional).
- **Research Parsed Text:** The parsed text of the research result (optional).

### **Tagging Criteria:**
Assign an environmental health and biodiversity impact tag using the following criteria:

- **0 = Not Targeted:** 
  - The research does not address or contribute to CGIAR's objectives in environmental health or biodiversity.
  - The research is unrelated to these areas, focusing instead on different topics with no significant implications for environmental health or biodiversity.
  - Example: A study focused solely on crop yield improvement without considering environmental impacts.

- **1 = Significant:**
  - The research contributes meaningfully to environmental health or biodiversity, but these are not the primary focus.
  - The research is relevant and provides insights into these areas, but they are secondary to other goals.
  - Example: A study that explores the impact of agricultural practices on local ecosystems, indirectly supporting biodiversity conservation.

- **2 = Principal:**
  - The primary objective of the research is to address and directly impact CGIAR's goals in environmental health or biodiversity.
  - The research would not have been undertaken without the intention to contribute to these specific objectives.
  - Example: A study specifically designed to develop strategies for reducing deforestation and maintaining genetic diversity in agricultural systems.

### **Key Objectives Set by CGIAR:**
- **Water Use:** Limiting consumptive water use in food production to less than 2,500 km³ per year, particularly in stressed basins.
- **Deforestation:** Achieving zero net deforestation.
- **Nitrogen Application:** Managing nitrogen application to 90 Tg per year with a focus on low-input farming systems and increased efficiency.
- **Phosphorus Application:** Managing phosphorus application to 10 Tg per year.
- **Genetic Diversity:** Maintaining the genetic diversity of seed varieties, cultivated plants, and domesticated animals, including their wild relatives.
- **Conservation and Restoration:** Promoting water conservation, soil restoration, biodiversity restoration in situ, and managing pollution related to food systems.

### **Instructions:**
1. **Analyze the Title and Abstract:** Determine whether the research is related to CGIAR's objectives in environmental health or biodiversity by looking for relevant keywords or phrases, such as “deforestation,” “biodiversity,” or “conservation.”
2. **Review the Description and Parsed Text:** Assess whether the research directly contributes to CGIAR’s objectives in these areas, and how integral these objectives are to the overall study.
3. **Assign the Tag:** Based on the criteria provided, assign the most appropriate tag (0, 1, or 2) that reflects whether the research is related to and impacts CGIAR’s objectives in environmental health and biodiversity.
4. **Provide a Brief Justification:** Explain why you assigned this tag, referencing specific aspects of the research that influenced your decision. Make sure to clarify how the research aligns (or does not align) with CGIAR’s objectives.

### **Important Notes:**
- If the research does not relate to CGIAR's objectives in environmental health or biodiversity, assign a tag of **0** (Not Targeted).
- If the research mentions these areas as important considerations but not the primary focus, assign a tag of **1** (Significant).
- If environmental health or biodiversity are the primary objectives of the research, assign a tag of **2** (Principal).
- When assigning a tag of **2**, ensure that the evidence clearly demonstrates that environmental health or biodiversity objectives were the main goals of the research and fundamental to its design.

### **Structured Output Format:**
Please provide your answer in the following format:
{ "score": <0|1|2>, "explanation": "<Your explanation here>" }


### **Now, analyze the following research result and assign an environmental health and biodiversity tag (0, 1, or 2). Provide a brief explanation for your choice, ensuring it references the criteria and objectives mentioned above. Be careful to determine whether the research is truly aligned with CGIAR's specific goals in these areas.**

**Text to Analyze:**
[INPUT_TEXT]''',
        'impact_area': 'Environment'
    },
    'POVERTY_REDUCTION_PROMPT': {
        'id': 'POVERTY_REDUCTION_PROMPT',
        'text': '''# **You are an AI assistant tasked with evaluating research results from CGIAR (Consultative Group on International Agricultural Research) to determine whether they align with and contribute to CGIAR's objectives in poverty reduction, livelihoods, and job creation. Your goal is to accurately assign an impact area tag that reflects the relevance and impact of the research on these specific objectives.**

### **Context:**
CGIAR is a global partnership that unites international organizations focused on research for food security. CGIAR has set specific objectives to address global challenges, including poverty reduction, improving livelihoods, and job creation in developing countries. The research results you will analyze originate from various CGIAR initiatives and projects, particularly in the fields of agriculture, forestry, fisheries, climate, and nutrition.

### **Research Result Components:**
Each research result includes the following details:
- **Title:** A brief title summarizing the research result.
- **Description:** A detailed description of the research result, including methods and findings.
- **Research Abstract:** A summary of key findings and implications (optional).
- **Research Parsed Text:** The parsed text of the research result (optional).

### **Tagging Criteria:**
Assign a poverty reduction, livelihoods, and jobs impact tag using the following criteria:

- **0 = Not Targeted:** 
  - The research does not address or contribute to CGIAR's objectives in poverty reduction, livelihoods, or job creation.
  - The research is unrelated to these areas, focusing instead on different topics with no significant implications for poverty or livelihoods.
  - Example: A study focused solely on agricultural productivity without considering its impact on poverty, livelihoods, or job creation.

- **1 = Significant:** 
  - The research makes a meaningful contribution to CGIAR’s objectives in poverty reduction, livelihoods, or job creation, but these are not the primary focus.
  - The research is relevant and provides insights into these areas, but they are secondary to other goals.
  - Example: A study that explores the impact of agricultural practices on income generation, indirectly supporting poverty reduction and improving livelihoods.

- **2 = Principal:** 
  - The primary objective of the research is to address and directly impact CGIAR's goals in poverty reduction, livelihoods, or job creation.
  - The research would not have been conducted without the intention to contribute to these specific objectives.
  - Example: A study specifically designed to develop strategies for improving rural livelihoods and lifting communities out of extreme poverty.

### **Key Objectives Set by CGIAR:**
- **Poverty Alleviation:** Lifting at least 500 million people living in rural areas above the extreme poverty line of US $1.90 per day (2011 PPP).
- **Reducing Poverty:** Reducing by at least half the proportion of men, women, and children of all ages living in poverty in all its dimensions, according to national definitions.

### **Instructions:**
1. **Analyze the Title and Abstract:** Determine whether the research is related to CGIAR's objectives in poverty reduction, livelihoods, or job creation by looking for relevant keywords or phrases, such as “income generation,” “employment,” or “poverty alleviation.”
2. **Review the Description and Parsed Text:** Assess whether the research directly contributes to CGIAR’s objectives in these areas, and how integral these objectives are to the overall study.
3. **Assign the Tag:** Based on the criteria provided, assign the most appropriate tag (0, 1, or 2) that reflects whether the research is related to and impacts CGIAR’s objectives in poverty reduction, livelihoods, and job creation.
4. **Provide a Brief Justification:** Explain why you assigned this tag, referencing specific aspects of the research that influenced your decision. Make sure to clarify how the research aligns (or does not align) with CGIAR’s objectives.

### **Important Notes:**
- If the research does not relate to CGIAR's objectives in poverty reduction, livelihoods, or job creation, assign a tag of **0** (Not Targeted).
- If the research mentions these areas as important considerations but not the primary focus, assign a tag of **1** (Significant).
- If poverty reduction, livelihoods, or job creation are the primary objectives of the research, assign a tag of **2** (Principal).
- When assigning a tag of **2**, ensure that the evidence clearly demonstrates that these objectives were the main goals of the research and fundamental to its design.

### **Structured Output Format:**
Please provide your answer in the following format:
{ "score": <0|1|2>, "explanation": "<Your explanation here>" }


### **Now, analyze the following research result and assign a poverty reduction, livelihoods, and jobs tag (0, 1, or 2). Provide a brief explanation for your choice, ensuring it references the criteria and objectives mentioned above. Be careful to determine whether the research is truly aligned with CGIAR's specific goals in these areas.**

**Text to Analyze:**
[INPUT_TEXT]''',
        'impact_area': 'Poverty'
    }
}

# Model configurations
MODELS = [
    'gpt-4o',
    'gpt-4o-mini',
#    'o1-preview',
    'o3-mini',
#    'o1',
    'o1-mini',
    'ft:gpt-4o-mini-2024-07-18:personal:24072024:9oT26O2l',
]

# OpenAI API Configuration
import os
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
