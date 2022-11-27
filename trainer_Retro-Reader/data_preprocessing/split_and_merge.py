import json
from collections import OrderedDict


def split_and_merge_dataset(unans_path, span_path, how_path, n, train_data=True):
    with open(unans_path, 'r', encoding='utf-8') as f:
        unans_data = json.load(f)
        unans_batch = int(len(unans_data['data']) // n)
    with open(span_path, 'r', encoding='utf-8') as f:
        span_data = json.load(f)
        span_batch = int(len(span_data['data']) // n)
    with open(how_path, 'r', encoding='utf-8') as f:
        how_data = json.load(f)
        how_batch = int(len(how_data['data']) // n)

    newf = OrderedDict()
    newf['version'] = "v2.0"
    newf['data'] = []

    span_batch = span_batch // 10
    how_batch = how_batch // 10
    unans_batch = unans_batch // 10

    span_start, span_end = 0, span_batch
    how_start, how_end = 0, how_batch
    unans_start, unans_end = 0, unans_batch

    for _ in range(10):

        for i in range(span_start, span_end):
            title = span_data['data'][i]['doc_title']

            for k in range(len(span_data['data'][i]['paragraphs'])):
                temp_list = []

                for j in range(len(span_data['data'][i]['paragraphs'][k]['qas'])):
                    question = span_data['data'][i]['paragraphs'][k]['qas'][j]['question']
                    ids = span_data['data'][i]['paragraphs'][k]['qas'][j]['question_id']
                    answer_text = span_data['data'][i]['paragraphs'][k]['qas'][j]['answers']['text']
                    answer_start = span_data['data'][i]['paragraphs'][k]['qas'][j]['answers']['answer_start']  # null
                    temp = {'question': question,
                            'id': ids,
                            'answers': [
                                {'text': answer_text,
                                 'answer_start': answer_start}],
                            'is_impossible': False}
                    temp_list.append(temp)

                context = span_data['data'][i]['paragraphs'][k]['context']

            newf['data'].append({'title': title, 'paragraphs': [{'qas': temp_list, 'context': context}]})

        span_start += span_batch
        span_end += span_batch

        for i in range(unans_start, unans_end):
            title = unans_data['data'][i]['doc_title']
            for k in range(len(unans_data['data'][i]['paragraphs'])):
                temp_list = []
                for j in range(len(unans_data['data'][i]['paragraphs'][k]['qas'])):
                    plausible_answers = unans_data['data'][i]['paragraphs'][k]['qas'][j]['answers']['text']
                    answer_start = unans_data['data'][i]['paragraphs'][k]['qas'][j]['answers']['answer_start']
                    question = unans_data['data'][i]['paragraphs'][k]['qas'][j]['question']
                    ids = unans_data['data'][i]['paragraphs'][k]['qas'][j]['question_id']

                    temp = {'plausible_answers': [{'text': plausible_answers, 'answer_start': answer_start}],
                            'question': question,
                            'id': ids,
                            'answers': [],
                            'is_impossible': True}

                    temp_list.append(temp)

                context = unans_data['data'][i]['paragraphs'][k]['context']

            newf['data'].append({'title': title, 'paragraphs': [{'qas': temp_list, 'context': context}]})

        unans_start += unans_batch
        unans_end += unans_batch

        for i in range(how_start, how_end):
            title = how_data['data'][i]['doc_title']

            for k in range(len(how_data['data'][i]['paragraphs'])):
                temp_list = []

                for j in range(len(how_data['data'][i]['paragraphs'][k]['qas'])):
                    question = how_data['data'][i]['paragraphs'][k]['qas'][j]['question']
                    ids = how_data['data'][i]['paragraphs'][k]['qas'][j]['question_id']
                    answer_text = how_data['data'][i]['paragraphs'][k]['qas'][j]['answers']['text']
                    answer_start = how_data['data'][i]['paragraphs'][k]['qas'][j]['answers']['answer_start']  # null

                    temp = {'question': question,
                            'id': ids,
                            'answers': [
                                {'text': answer_text,
                                 'answer_start': answer_start}],
                            'is_impossible': False}

                    temp_list.append(temp)

                context = how_data['data'][i]['paragraphs'][k]['context']

            newf['data'].append({'title': title, 'paragraphs': [{'qas': temp_list, 'context': context}]})

        how_start += how_batch
        how_end += how_batch

    if n == 1:
        f_num = 18
    elif n == 1.5:
        f_num = 12
    elif n == 3:
        f_num = 8
    elif n == 5:
        f_num = 4

    if train_data:
        f_name = "training"
    else:
        f_name = "validation"

    with open(f'./{f_name}_{f_num}m.json', 'w', encoding='utf-8') as make_file:
        json.dump(newf, make_file, ensure_ascii=False, indent=2)
