import openai
import requests
import time
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from rich.console import Console
import nums_from_string
from langchain.document_loaders import PyPDFLoader
import os
from langchain.prompts import PromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from getpass import getpass
from langchain.chains.conversation.memory  import ConversationBufferMemory
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate

import pandas as pd
import datetime
import pytz
from tiktoken import get_encoding
import sqlite3
# from keys import apikey
# from keys import TOKEN

from dotenv import load_dotenv
# from conexionsqlite import  *
console = Console()

# ------------------------------------- API KEY OPENAI -------------------------------------- #
load_dotenv()
apikey = os.getenv('apikey')
openai.api_key = apikey#apikey
os.environ['OPENAI_API_KEY'] = apikey
TOKEN = os.getenv('TOKEN')
# 



# ----- Info PRoducto
#@markdown info prod
#-********************************** NOTAS *********** -****************- ***************
# Verificar que antes de activar cualquier plan primero se sepa cual es el plan elegido de el cliente.
template = """
La siguiente es una conversación entre un humano y una inteligencia artificial llamada Aura.

Esta IA se llama Aura y es un asistente de apoyo emocional con amplio conocimiento en psicológia.
Aura es muy amable, comprensiva, empatica y solidaria siempre.
Si el humano saluda a Aura, Aura saluda y se presenta como Aura, y pregunta cómo se siente el Humano e indaga el por qué, iniciando una conversación.
Aura pregunta el nombre del Humano.
Aura tiene la tarea de indagar información del Humano, debe averiguar sobre el duelo emocional en que se encuentre el Humano y aconsejarlo en su duelo.

Aura nunca deja acabar la conversación, Aura pregunta si quiere que hablen de algún tema y sostiene la conversación.

Si el Humano indica que quiere recibir ayuda profesional de un psicologo o psiquiatra, Aura solicita Nombre y número de celular, e indica que en unos minutos un psicologo se pondrá en contacto.    
Si Aura considera que el Humano debe recibir ayuda profesional de un psicologo o psiquiatra, Aura solicita Nombre y número de celular.    

Aura utiliza muchos emojis.


Conversación actual:  {history}
Humano: {input}

Aura:

"""

PLANTILLA = PromptTemplate(
    input_variables=["history", "input"], template=template
)

tokens_plantilla = len(get_encoding("cl100k_base").encode(template))
# Funciones



######################################## FUNCIONES ######################################### 

def get_updates(offset):
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    #https://api.telegram.org/bot6596961934:AAGTURlsHdNfrDXqSMBIEqnVYhxGujlhaH0/getUpdates
    # 6596961934:AAGTURlsHdNfrDXqSMBIEqnVYhxGujlhaH0
    params = {"timeout": 100, "offset": offset}
    response = requests.get(url, params=params)
    return response.json()["result"]

def send_messages(chat_id, text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    params = {"chat_id": chat_id, "text": text}
    response = requests.post(url, params=params)
    return response

def almacenar_conversacion(dic_memory, id,chat_gpt3_5,tokens_plantilla,limite_tokens,max_token_limit_memory,falla_memoria):
    print("* Almacenando en memoria *")
    id=str(id)
    print(f"AlmacenandoID: {id} en historial... {len(dic_memory)}")
    
    if id in dic_memory:
        
        if dic_memory[id]['counter_tokens'] > limite_tokens :
            del dic_memory[id]
            falla_memoria =True
            dic_memory,falla_memoria = almacenar_conversacion(dic_memory, id,               
                                                        chat_gpt3_5,
                                                        tokens_plantilla,   limite_tokens,max_token_limit_memory,
                                                        falla_memoria
                                                        )
            
        
    else: 
        dic_memory[id] = { 
                            "chain": ConversationChain( llm=chat_gpt3_5, 
                                            memory=ConversationTokenBufferMemory(#ConversationBufferMemory( #ConversationSummaryBufferMemory(llm=OpenAI(),k=4)
                                                llm=OpenAI(),
                                                # max_history = 6,
                                                max_token_limit = max_token_limit_memory),
                                            verbose=False,
                                            prompt = PLANTILLA
                                            ),
                            
                            # Prompt Token Counter to not exceed the limit
                            "counter_tokens":0,
                          
                            # Input token count to estimate cost. Human
                            "input_tokens":0,
                            
                            # Output token count to estimate cost. Model
                            "output_tokens":0,
                            
                            # Costos TOTALES 
                            "total_inputs_cost":  0,             
                            "total_outputs_cost":  0
        }
        
              
    # print("valor:",dic_memory[id])
    return dic_memory,falla_memoria#dic_memory

def fecha_hora():
    zona_horaria_colombia = pytz.timezone('America/Bogota')
    hora_actual_colombia = datetime.datetime.now(zona_horaria_colombia)

    # Formatea la hora en un formato legible
    fecha_hora_formateada = hora_actual_colombia.strftime('%Y-%m-%d %H:%M:%S')

    # Imprime la hora en Colombia formateada
    print(f"----------------- {fecha_hora_formateada} -----------------")
    return fecha_hora_formateada

def main(falla_memoria=False):
    # try:
        print("Starting bot...")


        # mensajes=[]
        offset = 0
        count = 0
        COSTO_TOTAL = 0
        token_count_memory = 0
        tokens_user = 0
        tokens_ia = 0
        cost_input_model =0.0015/1000 #usd/ 1K tokens gpt-3.5-turbo
        cost_output_model = 0.002/1000 #usd/ 1K tokens gpt-3.5-turbo
        
        max_tokens_limit_user = 187
        max_token_limit_memory = 5000
        max_tokens_completion = 200
        offset_prevention = 0
        
        limite_tokens = 4097 - max_tokens_completion  -offset_prevention   #dic_memory[id]['counter_tokens'] gpt-3.5-turbo 4,097 tokens, para que se accione antes de generar error
        print(f"Limite de tokens por prompt: {limite_tokens} tokens")
        
        
        
        dic_memory = {} # {"<id>":[memory, sum_prompt_tokens, cost]}
        df = pd.DataFrame(
            columns=['Id','date','time','username','first_name','last_name','Mensaje','IA_rta'])
        tiempo_ON = fecha_hora() 
        tokens = tokens_plantilla
        chat_gpt3_5 = ChatOpenAI(
            openai_api_key=apikey,
            temperature=0.2,
            model='gpt-3.5-turbo',#'gpt-4',
            max_tokens=max_tokens_completion,
        )   
                
        while True: 
            print('.')              
            updates = get_updates(offset)
            
            if updates:
                
                tiempo = fecha_hora()
                print(f"Interacción N°: {count}")
                print(f"Conversaciones: {len(dic_memory)}")
                # print(f"Tokens: {tokens} {datetime.datetime.now(pytz.timezone('America/Bogota')).time().strftime('%H:%M:%S')}")
                
                for update in updates:
                    offset = update["update_id"] + 1
                    try:
                        

                        chat_id = str(update["message"]["chat"]['id'])
                        user_message = update["message"]["text"]
                        
                        try:
                            date = update["message"]['date']
                        except: date = "nan"
                        try:
                            username= update["message"]["from"]['username']
                        except: username = "nan"
                        
                        try:
                            first_name = update["message"]["from"]['first_name']
                        except: first_name = "nan"
                        try:
                            last_name = update["message"]["from"]['last_name']
                        except: last_name = "nan" 
                        
                    except:
                        chat_id = str(update["edited_message"]["chat"]['id'] )    
                        user_message = update["edited_message"]["text"]
                        
                        try: date = update["edited_message"]['date']
                        except: date = "nan"
                        
                        try: username= update["edited_message"]["from"]['username']
                        except: username = "nan"
                        
                        try:first_name = update["edited_message"]["from"]['first_name']
                        except: first_name = "nan"
                        
                        try:last_name = update["edited_message"]["from"]['last_name']
                        except: last_name = "nan" 
                        
                    
                    tokens_user = int(len(get_encoding("cl100k_base").encode(user_message)))
                    
                    if tokens_user < max_tokens_limit_user:
                        if chat_id in dic_memory:
                            
                            token_count_memory = dic_memory[chat_id]['input_tokens'] + dic_memory[chat_id]['output_tokens']
                            
                            if token_count_memory>max_token_limit_memory:
                                token_count_memory = max_token_limit_memory
                                                                    # por ahora no considero el numeoro exacto de tokens en memoria memory.chat_memory.get_token_count()
                            dic_memory[chat_id]['counter_tokens'] = tokens_user + tokens_plantilla + token_count_memory # Igual porque es el contador de tokens del prompt
                                                                                                    # el cual utilizo para no exeder el límite
                                                                                                    
                        dic_memory,falla_memoria = almacenar_conversacion(dic_memory, chat_id,               
                                                            chat_gpt3_5,
                                                            tokens_plantilla,   limite_tokens ,max_token_limit_memory,
                                                            falla_memoria
                                                            )
                        dic_memory[chat_id]['counter_tokens'] = tokens_user + tokens_plantilla + token_count_memory
                    else:pass   
                        
                    
                    print(f"User {username} | Received message: {user_message}")
                    # print(dic_memory)
                    # conversacion = dic_memory[chat_id]
                    if (falla_memoria==False) & (tokens_user < max_tokens_limit_user):
                        
                        r = dic_memory[chat_id]['chain'].predict(input=user_message)
                        
                        tokens_ia = int(len(get_encoding("cl100k_base").encode(r)))
                       
                        dic_memory[chat_id]['input_tokens']+=tokens_user
                        dic_memory[chat_id]['output_tokens']+=tokens_ia
                        
                        actual_message_imput_cost =  (tokens_user+tokens_plantilla+token_count_memory)*cost_input_model
                        actual_message_output_cost = tokens_ia*cost_output_model
                        tokens_totales = tokens_user+tokens_plantilla+token_count_memory + tokens_ia
                        dic_memory[chat_id]['total_inputs_cost']+=actual_message_imput_cost
                        dic_memory[chat_id]['total_outputs_cost']+=actual_message_output_cost
                        
                        COSTO_TOTAL+=actual_message_imput_cost+actual_message_output_cost
                        
                        # print(f"Conversaciones Almacenadas: {len(dic_memory)}\n")
                        print(f"\n--------- Tokens y Costos Aproximados | Usuario: {username} ----------\n")
                        print(f"Tokens aprox en memoria: {token_count_memory}")
                        print(f"Tokens totales en buffer: {int(len(get_encoding('cl100k_base').encode(str(dic_memory[chat_id]['chain'].memory.buffer))))}")
                        print("Inputs:")
                        print(f" Costo Input: {round(actual_message_imput_cost,4)} USD, por {dic_memory[chat_id]['counter_tokens']} Tokens") # (tok_template+tok_memory+token_messages) * input_cost
                        print(f" Costo Total Inputs: {round(dic_memory[chat_id]['total_inputs_cost'],4)} USD")
                        print("Outputs:")
                        print(f" Costo Output: {round(actual_message_output_cost,4)} USD por {tokens_ia} Tokens")
                        print(f" Costo Total Output: {round(dic_memory[chat_id]['total_outputs_cost'],4)} USD")
                        print("Acumulado:")
                        print(f"Costo Acumulado del Usuario: {round(dic_memory[chat_id]['total_inputs_cost']+dic_memory[chat_id]['total_outputs_cost'],2)} USD\n")
                        print("-------------------------------------------------------------------------")
                        print(f"         COSTO TOTAL ACUMULADO: {round(COSTO_TOTAL,4)} USD")
                        print("-------------------------------------------------------------------------\n")
                        
                        # print(f"Tokens aproximados en memoria: {dic_memory[chat_id][1]}")
                    elif tokens_user > max_tokens_limit_user:
                        print(f"********** {tiempo}  : Límite de tokens de usuario superado ********")
                        r="Oh, parece que tu mensaje es demasiado extenso.📝 Para ofrecerte la mejor asistencia, sería genial si pudieras resumirlo o hacerme una pregunta más concisa.😊 Estoy aquí para ayudarte 💬"
                        tokens_user = 0
                    elif(falla_memoria==True):
                        print(f"********** {tiempo}  : Límite de tokens superado ********")
                        r="¡Ups! Parece que he tenido un pequeño fallo de memoria, ¡me disculpo por eso! 😅 ¿Puedes recordarme sobre qué estábamos hablando? Estoy aquí para ayudarte en lo que necesites."
                        dic_memory = {}
                        falla_memoria=False
                    
                    print(f"ai: {r}")
                    print('')
                    # if "salir123" in ia_rta.lower():
                    #     break 
                    
                    send_messages(chat_id, r)
                    
                    nuevo_registro = {'Id':str(chat_id),
                                    # 'date':date,
                                    'time':str(tiempo),
                                    'username':str(username),
                                    'first_name':str(first_name),
                                    'last_name':str(last_name),
                                    'Mensaje':str(user_message),
                                    'user_tokens': int(tokens_user),
                                    'IA_rta':str(r),
                                    'ia_tokens': int(tokens_ia),
                                    'memory_tokens':int(token_count_memory)
                                    }
                    
                    # lista_registro = [valor for valor in nuevo_registro.values()]
                    # print(str(tuple(nuevo_registro.values())),tuple(nuevo_registro.values()))
                    # cargar_registro_en_BD(bd="BOT_3.db",registro=tuple(nuevo_registro.values()))
                    
                    df = pd.concat([df,pd.DataFrame(nuevo_registro, index=[count])])
                    count+=1
                    # df, M.append(nuevo_registro,ignore_index=True)
                    if (len(df)>=5) & (len(df)%5==0):
                        aux= tiempo_ON.replace(' ','_').replace(':','').replace('-','_')
                        # aux= aux.replace(':','')
                        # aux= aux.replace('-','_')
                        df.to_excel(f"./hist/historial_completo_{aux}.xlsx")
            else:
                time.sleep(1)
    # except:
    #     main(falla_memoria=True)
        
        
if __name__ == '__main__':
    main()