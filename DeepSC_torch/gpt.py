import json
import re

import google
import google.generativeai as genai
import nltk

import time

def load_json2list():
  with open('../dataset/train-v2.0.json') as f:
    json_object = json.load(f)

  title_idx = 0
  paragraph_idx = 0
  context_id = 0
  context_list = []
  qas_list = []

  while True:
    try:
      title_itr = json_object['data'][title_idx]['title']
      context_list.append([])
      qas_list.append([])
    except IndexError:
      break
    while True:
      try:
        context_itr = json_object['data'][title_idx]['paragraphs'][paragraph_idx]['context']
        sentences = nltk.sent_tokenize(context_itr)

        context_list[title_idx].append([])
        for context_sen_itr in sentences:
          if len(context_sen_itr) > 0:
            context_list[title_idx][paragraph_idx].append({'title': title_itr, 'context_id': context_id, 'context': context_sen_itr})

        qas_list[title_idx].append([])
        for qas_itr in range(len(json_object['data'][title_idx]['paragraphs'][paragraph_idx]['qas'])):
          try:
            qas_list[title_idx][paragraph_idx].append(
              {'title': title_itr,
               'context_id': context_id,
               'question': json_object['data'][title_idx]['paragraphs'][paragraph_idx]['qas'][qas_itr]['question'],
               'answer': json_object['data'][title_idx]['paragraphs'][paragraph_idx]['qas'][qas_itr]['answers'][0]['text'],
               'is_impossible': json_object['data'][title_idx]['paragraphs'][paragraph_idx]['qas'][qas_itr]['is_impossible']
               }
            )
          except IndexError:
            qas_list[title_idx][paragraph_idx].append(
              {'title': title_itr,
               'context_id': context_id,
               'question': json_object['data'][title_idx]['paragraphs'][paragraph_idx]['qas'][qas_itr]['question'],
               'answer': '',
               'is_impossible': json_object['data'][title_idx]['paragraphs'][paragraph_idx]['qas'][qas_itr]['is_impossible']
               }
            )

      except IndexError:
        break
      paragraph_idx += 1
      context_id += 1

    title_idx += 1
    paragraph_idx = 0

  return context_list, qas_list

def ask_question(chat, q_query):
  while True:
    try:
      time.sleep(1)
      response = chat.send_message(q_query)
      impossible_check_query = "Did you say it is impossible to answer the previous question? Answer with Yes or No."
      time.sleep(1)
      impossible_response = chat.send_message(impossible_check_query)
      impossible_check = impossible_response.text == "Yes"
      harm_check = False
      break
    except google.api_core.exceptions.InternalServerError:
      pass
    except google.generativeai.types.generation_types.StopCandidateException:
      print("Harmful content detected!")
      response = None
      impossible_check = None
      harm_check = True
      break
    except google.generativeai.types.generation_types.BlockedPromptException:
      print("Harmful content detected!")
      response = None
      impossible_check = None
      harm_check = True
      break

  return response, impossible_check, harm_check

def check_answer(a_ex, chat):
  while True:
    try:
      time.sleep(1)
      a_query = "True answer: " + a_ex + ". Did you get the answer right? Please answer with True or False."
      tf_response = chat.send_message(a_query)
      tf_check = tf_response.text
      harm_check = False
      break
    except google.api_core.exceptions.InternalServerError:
      pass
    except google.generativeai.types.generation_types.StopCandidateException:
      print("Harmful content detected !")
      tf_response = None
      tf_check = None
      harm_check = True
      break
    except google.generativeai.types.generation_types.BlockedPromptException:
      print("Harmful content detected !")
      tf_response = None
      tf_check = None
      harm_check = True
      break

  return tf_response, tf_check, harm_check

if __name__ == "__main__":
  genai.configure(api_key='AIzaSyAIDGGnKf4NFyGFPSIrpyIV6O5thj85RR4')
  model = genai.GenerativeModel('gemini-pro')

  context_list, qas_list = load_json2list()

  true_answer = 0
  false_answer = 0

  wrong_answer_list = []

  for title_id in range(len(qas_list)):
    for context_id in range(len(qas_list[title_id])):
      context_ex = ''
      for i in range(len(context_list[title_id][context_id])):
        context_ex += context_list[title_id][context_id][i]['context']
        context_ex += " "

      for qas_id in range(len(qas_list[title_id][context_id])):
        chat = model.start_chat(history=[])

        # title_id = 4
        # context_id = 0
        # qas_id = 5

        q_ex = qas_list[title_id][context_id][qas_id]['question']
        a_ex = qas_list[title_id][context_id][qas_id]['answer']
        is_impossible_ex = qas_list[title_id][context_id][qas_id]['is_impossible']

        q_query = "Title: " + context_list[title_id][context_id][0]['title'] + \
                  ". Context: " + context_ex + \
                  "Based on the above context paragraph, please give or infer an answer simply about the following question. Question: " + q_ex

        response, impossible_check, harm_check = ask_question(chat, q_query)

        if harm_check:
          wrong_answer_list.append(
            {'index': [title_id, context_id, qas_id], 'query': q_query}
          )
          continue

        if impossible_check:
          if is_impossible_ex:
            # print("The question is really impossible to answer.")
            true_answer += 1
          else:
            # print("True answer: " + a_ex)
            false_answer += 1
            wrong_answer_list.append(
              {'index': [title_id, context_id, qas_id], 'response': response.text}
            )
        else:
          tf_response, tf_check, harm_check = check_answer(a_ex, chat)

          if harm_check:
            wrong_answer_list.append(
              {'index': [title_id, context_id, qas_id], 'response': response.text, 'answer': a_ex}
            )
            continue

          if tf_check:
            true_answer += 1
          else:
            false_answer += 1
            wrong_answer_list.append(
              {'index': [title_id, context_id, qas_id], 'response': response.text, 'tf_response': tf_response.text}
            )

        print([title_id, context_id, qas_id], " True answer: ", true_answer, "False answer: ", false_answer)


  print("True answer: ", true_answer, "False answer: ",false_answer)



