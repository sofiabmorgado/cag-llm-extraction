"""Extract the FFR and iFR values from the reports using a LLM"""

from langchain_ollama import OllamaLLM
import pandas as pd 
import json
from tqdm import tqdm
tqdm.pandas()

OLLAMA_PROTOCOL="http"
OLLAMA_HOST="localhost"
OLLAMA_PORT="11434"


def extract_FFR_iFR(report, llm, question_type):
    """Extract the FFR and iFR"""
    
    question_zero_shot = """
    És um especialista em relatórios de angiografias/coronariografia e angioplastia.
    O relatório inclui informação relativa aos indices de fisiologia: fractional flow reserve ou FFR e instant wave-free ratio ou iFR.
    Retira do texto os valores corretos de FFR e iFR e completa o seguinte template em JSON com os valores corretos para cada artéria (quando o valor não está presente no relatório, mantém "null")
    
    Template:
    {
    "Tronco Comum": 
        {
            "FFR": null,
            "iFR": null
        },
    "Descendente Anterior":    
        {
            "FFR": null,
            "iFR": null
        },
    "Circunflexa":
        {
            "FFR": null,
            "iFR": null
        },
    "Coronária Direita":
        {
            "FFR": null,
            "iFR": null
        },
    "Outras artérias":
        {
            "FFR": null,
            "iFR": null
        }
    }
    
    Relatório:
    |""" + str(report) + """|
    
    Responde apenas com o template preenchido, sem incluir mais texto na tua resposta, e sem justificar. 
    A tua resposta deve ser um JSON válido.
    Para cada valor de FFR ou iFR do template, deves apenas colocar um número décimal (com . a separar as unidades das décimas, e não vírgula, como por exemplo, 0.91). 
    Nenhum outro tipo de informação deve ser preenchida, como palavras ou frases. 
    Nunca coloques um valor que não está presente no relatório. Em caso de dúvida, preenche com null.
    Quando há mais de uma medição de FFR ou iFR para a mesma atéria, inclui o valor mais baixo.
    """
    
    question_one_shot = """
    És um especialista em relatórios de angiografias/coronariografia e angioplastia.
    O relatório inclui informação relativa aos indices de fisiologia: fractional flow reserve ou FFR e instant wave-free ratio ou iFR.
    Retira do texto os valores corretos de FFR e iFR e completa o seguinte template em JSON com os valores corretos para cada artéria (quando o valor não está presente no relatório, mantém "null").
    
    Template:
    {
    "Tronco Comum": 
        {
            "FFR": null,
            "iFR": null
        },
    "Descendente Anterior":    
        {
            "FFR": null,
            "iFR": null
        },
    "Circunflexa":
        {
            "FFR": null,
            "iFR": null
        },
    "Coronária Direita":
        {
            "FFR": null,
            "iFR": null
        },
    "Outras artérias":
        {
            "FFR": null,
            "iFR": null
        }
    }
    
    
    De seguida tens um exemplo de como deves responde para o relatorio exemplo. 
    Repara como o valor de FFR para a coronária direita do exemplo se encontra explicitamentamente referido no texto. 
    
    Relatório Exemplo 1: 
    | 
    Pressão arterial sistémica normal. 
    Pressão telediastólica ventricular esquerda não avaliada. 
    CORONARIOGRAFIA: 
    Coronária esquerda: tronco comum sem lesões. Descendente anterior com excelente resultado da intervenção anterior. Circunflexa e marginais sem lesões. 
    Coronária direita com duas lesões descritas anteriormente de gravidade moderada, uma mais proximal e outra médio-distal. 
    VENTRICULOGRAFIA: 
    Não efectuada. 
    Avaliação funcional da coronária direita (FFR): 
    Afim de avaliar a gravidade das lesões da coronária direita foi efectuado FFR que mostrou que a lesão proximal era grave (FFR = 0.68) e que a distal não apresentava qualquer gradiente. 
    ANGIOPLASTIA: 
    Angioplastia da lesão proximal da coronária direita com implantação em primeira intenção de stent com everolimus XIENCE 2,75 x 8 mm a 14 ATM com bom resultado final.  
    Seguidamente voltou-se a efectuar FFR que confirmou ausência de gravidade da lesão mais distal e desaparecimento da estenose proximal.  
    Cateterismo efectuado pela ARD - 6Fr. 
    CONCLUSÃO: 
    Bom resultado da intervenção anterior sobre a descendente anterior. 
    Angioplastia de lesão proximal da coronária direita com colocação de stent com antiproliferativo e bom resultado final. |
        
    A tua resposta para o relatório exemplo seria:
    {
    "Tronco Comum": 
        {
            "FFR": null,
            "iFR": null
        },
    "Descendente Anterior":    
        {
            "FFR": null,
            "iFR": null
        },
    "Circunflexa":
        {
            "FFR": null,
            "iFR": null
        },
    "Coronária Direita":
        {
            "FFR": 0.68,
            "iFR": null
        },
    "Outras artérias":
        {
            "FFR": null,
            "iFR": null
        }
    }
    
    Relatório Exemplo 1: 
    | 
    Ventriculografia não realizada. 
    Tronco comum sem lesões significativa. 
    Estenose não significativa do ostio da descendente anterior. Foi realizada avaliação funcional da lesão com PressureWire (FFR de 0.92). 
    Stent colocados na descendente anterior sem reestenose. 
    Estenose não significativa da circunflexa proximal. Foi realizada avaliação funcional da lesão com PressureWire (FFR de 0.98). 
    Stent colocados na circunflexa sem reestenose. 
    Coronária direita não dominante, com doença difusa. 
    |
    
    A tua resposta para o relatório exemplo seria:
    {
    "Tronco Comum": 
        {
            "FFR": null,
            "iFR": null
        },
    "Descendente Anterior":    
        {
            "FFR": 0.92,
            "iFR": null
        },
    "Circunflexa":
        {
            "FFR": 0.98,
            "iFR": null
        },
    "Coronária Direita":
        {
            "FFR": null,
            "iFR": null
        },
    "Outras artérias":
        {
            "FFR": null,
            "iFR": null
        }
    }
    
    Relatório:
    |""" + str(report) + """|
    
    Responde apenas com o template preenchido, sem incluir mais texto na tua resposta, e sem justificar. 
    A tua resposta deve ser um JSON válido. 
    Nunca deves incluir valores que não estão presentes no relatório. Em caso de dúvida, preenche com null.
    Nenhum outro tipo de informação deve ser preenchida, como palavras ou frases. 
    Para cada valor de FFR ou iFR do template, deves apenas colocar um número décimal. 
    Quando há mais de uma medição de FFR ou iFR para a mesma atéria, inclui o valor mais baixo.
    """
    
    
    question_one_shot_absurd = """
    És um especialista em relatórios de angiografias/coronariografia e angioplastia.
    O relatório inclui informação relativa aos indices de fisiologia: fractional flow reserve ou FFR e instant wave-free ratio ou iFR.
    Retira do texto os valores corretos de FFR e iFR e completa o seguinte template em JSON com os valores corretos para cada artéria (quando o valor não está presente no relatório, mantém "null").
    
    Template:
    {
    "Tronco Comum": 
        {
            "FFR": null,
            "iFR": null
        },
    "Descendente Anterior":    
        {
            "FFR": null,
            "iFR": null
        },
    "Circunflexa":
        {
            "FFR": null,
            "iFR": null
        },
    "Coronária Direita":
        {
            "FFR": null,
            "iFR": null
        },
    "Outras artérias":
        {
            "FFR": null,
            "iFR": null
        }
    }
    
    
    De seguida tens um exemplo de como deves responde para o relatorio exemplo. 
    Repara como o valor de FFR para a coronária direita do exemplo se encontra explicitamentamente referido no texto. 
    
    Relatório Exemplo 1: 
    | 
    Pressão arterial sistémica normal. 
    Pressão telediastólica ventricular esquerda não avaliada. 
    CORONARIOGRAFIA: 
    Coronária esquerda: tronco comum sem lesões. Descendente anterior com excelente resultado da intervenção anterior. Circunflexa e marginais sem lesões. 
    Coronária direita com duas lesões descritas anteriormente de gravidade moderada, uma mais proximal e outra médio-distal. 
    VENTRICULOGRAFIA: 
    Não efectuada. 
    Avaliação funcional da coronária direita (FFR): 
    Afim de avaliar a gravidade das lesões da coronária direita foi efectuado FFR que mostrou que a lesão proximal era grave (FFR = 6,8) e que a distal não apresentava qualquer gradiente. 
    ANGIOPLASTIA: 
    Angioplastia da lesão proximal da coronária direita com implantação em primeira intenção de stent com everolimus XIENCE 2,75 x 8 mm a 14 ATM com bom resultado final.  
    Seguidamente voltou-se a efectuar FFR que confirmou ausência de gravidade da lesão mais distal e desaparecimento da estenose proximal.  
    Cateterismo efectuado pela ARD - 6Fr. 
    CONCLUSÃO: 
    Bom resultado da intervenção anterior sobre a descendente anterior. 
    Angioplastia de lesão proximal da coronária direita com colocação de stent com antiproliferativo e bom resultado final. |
        
    A tua resposta para o relatório exemplo seria:
    {
    "Tronco Comum": 
        {
            "FFR": null,
            "iFR": null
        },
    "Descendente Anterior":    
        {
            "FFR": null,
            "iFR": null
        },
    "Circunflexa":
        {
            "FFR": null,
            "iFR": null
        },
    "Coronária Direita":
        {
            "FFR": 6.8,
            "iFR": null
        },
    "Outras artérias":
        {
            "FFR": null,
            "iFR": null
        }
    }
    
    Relatório Exemplo 1: 
    | 
    Ventriculografia não realizada. 
    Tronco comum sem lesões significativa. 
    Estenose não significativa do ostio da descendente anterior. Foi realizada avaliação funcional da lesão com PressureWire (FFR de 9.2). 
    Stent colocados na descendente anterior sem reestenose. 
    Estenose não significativa da circunflexa proximal. Foi realizada avaliação funcional da lesão com PressureWire (FFR de 9.8). 
    Stent colocados na circunflexa sem reestenose. 
    Coronária direita não dominante, com doença difusa. 
    |
    
    A tua resposta para o relatório exemplo seria:
    {
    "Tronco Comum": 
        {
            "FFR": null,
            "iFR": null
        },
    "Descendente Anterior":    
        {
            "FFR": 9.2,
            "iFR": null
        },
    "Circunflexa":
        {
            "FFR": 9.8,
            "iFR": null
        },
    "Coronária Direita":
        {
            "FFR": null,
            "iFR": null
        },
    "Outras artérias":
        {
            "FFR": null,
            "iFR": null
        }
    }
    
    Relatório:
    |""" + str(report) + """|
    
    Responde apenas com o template preenchido, sem incluir mais texto na tua resposta, e sem justificar. 
    A tua resposta deve ser um JSON válido. 
    Nunca deves incluir valores que não estão presentes no relatório. Em caso de dúvida, preenche com null.
    Nenhum outro tipo de informação deve ser preenchida, como palavras ou frases. 
    Para cada valor de FFR ou iFR do template, deves apenas colocar um número décimal. 
    Quando há mais de uma medição de FFR ou iFR para a mesma atéria, inclui o valor mais baixo.
    """
    
    
    if question_type == "zero_shot":
        return llm.invoke(question_zero_shot)
    elif question_type == "one_shot":
        return llm.invoke(question_one_shot)
    return llm.invoke(question_one_shot_absurd)


def clean_results_JSON(result):
    """Clean the results to get a valid JSON"""
    if result.startswith("```json"):
        return result[7:-3].strip()
    return result.strip()


def format_FFR_iFR(results):
    """Format the results from a JSON with FFR and iFR to a dataframe with columns for each artery"""
    arteries = ["Tronco Comum",
                "Descendente Anterior",
                "Circunflexa",
                "Coronária Direita",
                "Outras artérias"]
    try:
        results = results.replace("NA", "null")
        data = json.loads(results)
        print(data)
        values = {}
        for artery in data:
            nome = artery.replace(" ", "_")
            if nome in [a.replace(" ", "_") for a in arteries]:
                values[f"{nome}_FFR"] = data[artery]["FFR"]
                values[f"{nome}_iFR"] = data[artery]["iFR"]
        return values
    except:
        values = {}
        for artery in arteries:
            nome = artery.replace(" ", "_")
            values[f"{nome}_FFR"] = "NA"
            values[f"{nome}_iFR"] = "NA"
        return values


def get_FFR_iFR(df, path, question_type):
    #Extract a JSON with the FFR and iFR values from the reports
    df["Results"] = df["Conclusões"].progress_apply(lambda report: extract_FFR_iFR(report, llm, question_type))

    #Clean results
    df["Results_clean"] = df["Results"].apply(clean_results_JSON)

    #Format results
    extracted_values = df["Results_clean"].apply(format_FFR_iFR).apply(pd.Series)

    # Concatenar as novas colunas com o DataFrame original
    df = pd.concat([df, extracted_values], axis=1)

    #Select columns
    columns = ["id", "Conclusões", "Tronco_Comum_FFR", "Descendente_Anterior_FFR", "Circunflexa_FFR", "Coronária_Direita_FFR", "Outras_artérias_FFR", "Tronco_Comum_iFR", "Descendente_Anterior_iFR", "Circunflexa_iFR", "Coronária_Direita_iFR", "Outras_artérias_iFR"]
    df = df[columns]

    #Save
    df.to_csv(path, index=False)


#RUN
#Input file
reports_df = pd.read_csv('data/reports_groundtruth.csv')[["id", "Conclusões"]]

#Choose model and query (question_type: zero_shot or one_shot or one_shot_absurd)
ollama_model = "gpt-oss:20b"
llm = OllamaLLM(base_url=f'{OLLAMA_PROTOCOL}://{OLLAMA_HOST}:{OLLAMA_PORT}',
                model = ollama_model,
                temperature=0)

print("Getting FFR/iFR")
question_type = "zero_shot"
get_FFR_iFR(reports_df, f"results/<model_name_folder>/FFR_iFR.csv", question_type)