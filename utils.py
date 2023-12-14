import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from tqdm import tnrange, tqdm

from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score


def add_special_tokens():
	""" Returns GPT2 tokenizer after adding separator and padding tokens """
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	special_tokens = {'pad_token':'<|pad|>','sep_token':'<|sep|>'}
	tokenizer.add_special_tokens(special_tokens)
	return tokenizer

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_seq(model, context, length, device, temperature=1, top_k=0, top_p=0.0):
    """ Generates a sequence of tokens 
        Args:
            model: gpt/gpt2 model
            context: tokenized text using gpt/gpt2 tokenizer
            length: length of generated sequence.
            device: torch.device object.
            temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():  
        # for _ in tnrange(length):
        for _ in tqdm(range(length)):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


def beam_search(model, context, length, beam_size, device, temperature=1):
    """ Generate sequence using beam search https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
        Args:
            model: gpt/gpt2 model
            context: tokenized text using gpt/gpt2 tokenizer
            length: length of generated sequence.
            beam_size: >=1 and <= total_no_of_tokens
            device: torch.device object.
            temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
    """
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    with torch.no_grad():  
        inputs = {'input_ids': context}
        outputs = model(**inputs) 
        next_token_logits = outputs[0][0, -1, :] / temperature
        next_token_probs = F.softmax(next_token_logits)
        scores, indices = torch.topk(next_token_probs, beam_size)
        indices = indices.tolist()
        sequences = [[c] for c in indices]
        # for _ in tnrange(length-1):
        for _ in tqdm(range(length)):
            logits = torch.zeros(beam_size*len(next_token_logits))
            for j in range(len(sequences)):
                new_generated = torch.cat((context,torch.tensor([sequences[j]], dtype=torch.long, device=device)),dim=1)
                inputs = {'input_ids': new_generated}
                outputs = model(**inputs) 
                next_token_logits = outputs[0][0, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits)
                start, stop = j*len(next_token_logits), (j+1)*len(next_token_logits)
                logits[start:stop] = scores[j]*next_token_probs
            scores, new_logits_indices = torch.topk(logits,beam_size)
            logits = (new_logits_indices%50259).tolist()
            for j in range(len(sequences)):
                sequences[j] = sequences[j]+[logits[j]]
    return scores, sequences

def calculate_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split())

def calculate_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores[0]  # 返回第一个样本的得分

def calculate_bertscore(reference, hypothesis):
    """
    计算BERTScore。
    
    :param reference: 参考文本。
    :param hypothesis: 生成文本。
    :return: 返回BERTScore的P, R, F1分数。
    """
    P, R, F1 = score([hypothesis], [reference], lang='en', model_type='bert-base-uncased')
    return P.item(), R.item(), F1.item()

def generate_beam_sample(data, tokenizer, model, num=1, length=100, beam_size=3, device=torch.device('cuda'), br_eval =  False):
    """ Generate summaries for "num" number of articles using beam search.
        Args:
            data = GPT21024Dataset object
            tokenizer = gpt/gpt2 tokenizer
            num = number of articles for which summaries has to be generated
    """
    data_len = len(data)

    for i in range(num):
        if br_eval==True:
            i = random.randint(0, data_len - 1)
        sample = data[i]
        idx = sample['sum_idx']
        context = sample['article'][:idx].tolist()
        summary = sample['article'][idx+1:][:100].tolist()
        scores, sequences = beam_search(model, context, length, beam_size, device)
        print('new_article', end='\n\n')
        print(tokenizer.decode(context[:-1]), end='\n\n')
        print('actual_summary', end='\n\n')
        print(tokenizer.decode(summary), end='\n\n')
        for i in range(len(sequences)):
            text = tokenizer.convert_ids_to_tokens(sequences[i],skip_special_tokens=True)
            text = tokenizer.convert_tokens_to_string(text)  
            print("generated_summary-{} and Score is {}.".format(i+1, scores[i]), end='\n\n')
            print(text, end='\n\n')
            if br_eval==True:
                bleu_score = calculate_bleu(tokenizer.decode(summary), text)
                rouge_score = calculate_rouge(tokenizer.decode(summary), text)
                bertscore_P, bertscore_R, bertscore_F1 = calculate_bertscore(tokenizer.decode(summary), text)
                print("BLEU Score:", bleu_score)
                print("ROUGE Score:", rouge_score)
                print("BERTScore - Precision: {}, Recall: {}, F1: {}".format(bertscore_P, bertscore_R, bertscore_F1))


def generate_sample(data, tokenizer, model, num=1, eval_step=False, length=100, temperature=1, top_k=10, top_p=0.5, device=torch.device('cuda'), br_eval =  False):
    """ Generate summaries for "num" number of articles.
        Args:
            data = GPT21024Dataset object
            tokenizer = gpt/gpt2 tokenizer
            model = gpt/gpt2 model
            num = number of articles for which summaries has to be generated
            eval_step = can be True/False, checks generating during evaluation or not
    """
    data_len = len(data)

    for i in range(num):
        if br_eval==True:
            i = random.randint(0, data_len - 1)

        sample = data[i]
        idx = sample['sum_idx']
        context = sample['article'][:idx].tolist()
        summary = sample['article'][idx+1:][:100].tolist()
        generated_text = sample_seq(model, context, length, device, temperature, top_k, top_p)
        generated_text = generated_text[0, len(context):].tolist()
        text = tokenizer.convert_ids_to_tokens(generated_text,skip_special_tokens=True)
        text = tokenizer.convert_tokens_to_string(text)
        if eval_step==False:
            print('new_article', end='\n\n')
            print(tokenizer.decode(context), end='\n\n')
            print("generated_summary", end='\n\n')
            print(text, end='\n\n')
            print('actual_summary', end='\n\n')
            print(tokenizer.decode(summary), end='\n\n')
        else:
            print(tokenizer.decode(context), end='\n\n')
            print("generated_summary", end='\n\n')
        if br_eval==True:
            reference = tokenizer.decode(summary)
            bleu_score = calculate_bleu(reference, text)
            rouge_score = calculate_rouge(reference, text)
            bertscore_P, bertscore_R, bertscore_F1 = calculate_bertscore(reference, text)
            # 输出结果
            print("BLEU Score:", bleu_score)
            print("ROUGE Score:", rouge_score)
            print("BERTScore - Precision: {}, Recall: {}, F1: {}".format(bertscore_P, bertscore_R, bertscore_F1))

def generate_sample_all(data, tokenizer, model, num=1, length=100, save_path=None, temperature=1, top_k=10, top_p=0.5, device=torch.device('cuda')):
    """
    Generate summaries for the first 'num' articles in the dataset, save them to a file,
    and compute average ROUGE and BERTScores.
    
    Args:
        data: GPT21024Dataset object
        tokenizer: gpt/gpt2 tokenizer
        model: gpt/gpt2 model
        num: number of articles for which summaries need to be generated
        device: torch.device object
        save_path: path to save the generated summaries
        temperature, top_k, top_p: parameters for text generation
    """
    """Generate summaries for first 'num' articles in the dataset and calculate average ROUGE and BERTScores."""
    rouge_scores = []
    bert_scores = []

    with open(save_path, 'w', encoding='utf-8') as file:
        for i in range(num):
            sample = data[i]
            idx = sample['sum_idx']
            context = sample['article'][:idx].tolist()
            summary = sample['article'][idx+1:][:100].tolist()
            generated_text = sample_seq(model, context, length, device, temperature, top_k, top_p)  # Assuming 50 is the desired length
            generated_text = generated_text[0, len(context):].tolist()
            text = tokenizer.convert_ids_to_tokens(generated_text, skip_special_tokens=True)
            text = tokenizer.convert_tokens_to_string(text)

            # Compute scores
            reference = tokenizer.decode(summary)
            rouge_score = calculate_rouge(reference, text)
            rouge_scores.append(rouge_score)
            bertscore_P, bertscore_R, bertscore_F1 = calculate_bertscore(reference, text)
            bert_scores.append((bertscore_P, bertscore_R, bertscore_F1))

            # Write to file
            file.write(f"Article {i+1}:\n")
            file.write(f"Generated Summary: {text}\n")
            file.write(f"Actual Summary: {reference}\n")
            file.write(f"ROUGE Score: {rouge_score}\n")
            file.write(f"BERTScore - Precision: {bertscore_P}, Recall: {bertscore_R}, F1: {bertscore_F1}\n")
            file.write("\n")
    total_rouge_1 = {'f': 0, 'p': 0, 'r': 0}
    total_rouge_2 = {'f': 0, 'p': 0, 'r': 0}
    total_rouge_l = {'f': 0, 'p': 0, 'r': 0}

    for score in rouge_scores:
        total_rouge_1['f'] += score['rouge-1']['f']
        total_rouge_1['p'] += score['rouge-1']['p']
        total_rouge_1['r'] += score['rouge-1']['r']

        total_rouge_2['f'] += score['rouge-2']['f']
        total_rouge_2['p'] += score['rouge-2']['p']
        total_rouge_2['r'] += score['rouge-2']['r']

        total_rouge_l['f'] += score['rouge-l']['f']
        total_rouge_l['p'] += score['rouge-l']['p']
        total_rouge_l['r'] += score['rouge-l']['r']

    # 计算平均ROUGE分数
    avg_rouge = {
        'rouge-1': {
            'f': total_rouge_1['f'] / len(rouge_scores),
            'p': total_rouge_1['p'] / len(rouge_scores),
            'r': total_rouge_1['r'] / len(rouge_scores)
        },
        'rouge-2': {
            'f': total_rouge_2['f'] / len(rouge_scores),
            'p': total_rouge_2['p'] / len(rouge_scores),
            'r': total_rouge_2['r'] / len(rouge_scores)
        },
        'rouge-l': {
            'f': total_rouge_l['f'] / len(rouge_scores),
            'p': total_rouge_l['p'] / len(rouge_scores),
            'r': total_rouge_l['r'] / len(rouge_scores)
        }
    }

    # 计算平均BERTScore
    avg_bert = np.mean(bert_scores, axis=0)

    # 打印平均分数
    print("Average ROUGE-1: {}".format(avg_rouge['rouge-1']))
    print("Average ROUGE-2: {}".format(avg_rouge['rouge-2']))
    print("Average ROUGE-L: {}".format(avg_rouge['rouge-l']))
    print("Average BERTScore - Precision: {}, Recall: {}, F1: {}".format(avg_bert[0], avg_bert[1], avg_bert[2]))

    # 返回平均分数
    return {'rouge': avg_rouge, 'bert': avg_bert}
    # 初始化累加器
    total_rouge_1_f = total_rouge_2_f = total_rouge_l_f = 0
    total_rouge_1_p = total_rouge_2_p = total_rouge_l_p = 0
    total_rouge_1_r = total_rouge_2_r = total_rouge_l_r = 0

    # 累加每个评分
    for score in rouge_scores:
        total_rouge_1_f += score['rouge-1']['f']
        total_rouge_1_p += score['rouge-1']['p']
        total_rouge_1_r += score['rouge-1']['r']

        total_rouge_2_f += score['rouge-2']['f']
        total_rouge_2_p += score['rouge-2']['p']
        total_rouge_2_r += score['rouge-2']['r']

        total_rouge_l_f += score['rouge-l']['f']
        total_rouge_l_p += score['rouge-l']['p']
        total_rouge_l_r += score['rouge-l']['r']

    # 计算平均值
    avg_rouge_1_f = total_rouge_1_f / len(rouge_scores)
    avg_rouge_1_p = total_rouge_1_p / len(rouge_scores)
    avg_rouge_1_r = total_rouge_1_r / len(rouge_scores)

    avg_rouge_2_f = total_rouge_2_f / len(rouge_scores)
    avg_rouge_2_p = total_rouge_2_p / len(rouge_scores)
    avg_rouge_2_r = total_rouge_2_r / len(rouge_scores)

    avg_rouge_l_f = total_rouge_l_f / len(rouge_scores)
    avg_rouge_l_p = total_rouge_l_p / len(rouge_scores)
    avg_rouge_l_r = total_rouge_l_r / len(rouge_scores)

    # 打印平均分数
    print("Average ROUGE-1: F1={}, P={}, R={}".format(avg_rouge_1_f, avg_rouge_1_p, avg_rouge_1_r))
    print("Average ROUGE-2: F1={}, P={}, R={}".format(avg_rouge_2_f, avg_rouge_2_p, avg_rouge_2_r))
    print("Average ROUGE-L: F1={}, P={}, R={}".format(avg_rouge_l_f, avg_rouge_l_p, avg_rouge_l_r))
    # Calculate average scores
    avg_bert = np.mean(bert_scores, axis=0)
    print("Average BERTScore - Precision: {}, Recall: {}, F1: {}".format(avg_bert[0], avg_bert[1], avg_bert[2]))

    # Return average scores
    return avg_bert
