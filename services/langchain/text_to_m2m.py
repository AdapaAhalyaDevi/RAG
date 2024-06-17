from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.ollama import Ollama


RESPONSE_TEMPLATE = """
    **Executive Summary** : 
        here list of executive summary of meeting transcript point to point

    **Meeting Notes** :
        define meeting notes main key points with details

    **Other Key Points** :
        describe a key point of meeting

    **Tasks** :
        here define a list of task which is in transcript. like given task to any person, team etc.

    **Decisions** :
        create a list which about take a decision in transcript. 

    NOTE: all of the list separated by bullet points. if you not found about any topic related data in transcript then return None in that topic.
"""


PROMPT_TEMPLATE = """
    I Give a meeting transcript and that meeting transcript is conversion of meeting member. 

    Summarize the meeting transcript and generate detailed about meeting notes. The summary should include the following sections: Executive Summary, Meeting Notes, Other Key Points, Decisions, and Tasks & Action Items. 
    
    Give me each section is detailed and well-organized, using bullet points for clarity. Under 'Meeting Notes', create subsections for each major topic discussed, including detailed points and examples mentioned during the meeting.  

    "Meeting Transcript" : {transcript}

    For better understanding here is template for generate response and response must be like this : {template}
    """



def text_to_m2mquery(text: str, llm_model: str):

    prompt_template = PromptTemplate.from_template(
        template=PROMPT_TEMPLATE
        )

    llm = Ollama(model=llm_model)

    chain = prompt_template | llm | StrOutputParser()
    
    response_text = chain.invoke({"transcript": text, "template": RESPONSE_TEMPLATE})


    return {"response": response_text}