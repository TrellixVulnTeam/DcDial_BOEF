import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from transformers_copy import  OpenAIGPTTokenizer,OpenAIGPTLMHeadModel, GPT2Tokenizer
from transformers_copy.gpt2_model_stage2 import GPT2LMHeadModel

from utils import get_dataset, download_pretrained_model
MAX_INPUTS_LENGTH=512
MAX_HISTORY_LENGTH=200
MAX_ALL_HISTORY_LENGTH=256
MAX_ALL_KNOWLEDGE_LENGTH=400
max_tgt=75
max_src=75
max_srcPlustgt=150
max_srcPlusDis=150
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>","<pad>","<encoder>","<knl>","[CLS]"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>',"<encoder>","<knl>","[CLS]"]}

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def build_input_from_segments(knowledge, historys, historyKnl,reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2,pad,enc1,knl,cls= tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    #sequence = [[bos] + persona] + history + [reply + ([eos] if with_eos else [])]
    #sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    history=[]
    knowledge=knowledge[:256]
    count_src=0
    if len(historys[-1])>max_src:
        count_src+=1
        historys[-1]=historys[-1][:74]
    token_type_ids=[knl]*(len(knowledge)+1)
    token_type_ids+=[enc1]*(len(historys[-1])+1)
    token_type_ids1=[]
    history=[]
    for i in range(len(historys)-1):
        if(len(history+[speaker2 if i%2 else speaker1]+historys[-i-2])>MAX_HISTORY_LENGTH):
            break
        history=[speaker1 if (len(historys)-i)%2 else speaker2]+historys[-i-2]+history
        token_type_ids1=[speaker1 if (len(historys)-i)%2 else speaker2]*(len(historys[-i-2])+1)+token_type_ids1
    token_type_ids+=token_type_ids1
    if len(historys)%2:
        input_ids=[bos]+knowledge+[speaker1]+historys[-1]+history+[speaker1]+historys[-1]+[speaker2]+reply
        token_type_ids+=[speaker1]*(len(historys[-1])+1)
        token_type_ids+=[speaker2]*(len(reply)+1)
    else:
        input_ids=[bos]+knowledge+[speaker2]+historys[-1]+history+[speaker2]+historys[-1]+[speaker1]+reply
        token_type_ids+=[speaker2]*(len(historys[-1])+1)
        token_type_ids+=[speaker1]*(len(reply)+1)
    history2=[]
    for i in range(len(historys)-1):
        history2=history2+historys[i]
    instance = {}

    assert len(input_ids)==len(token_type_ids), '长度不相等'
    instance["input_ids"] = input_ids
    instance["token_type_ids"] = token_type_ids
    #instance["mc_token_ids"] = len(input_ids)-1
    knl_ids=-len([speaker1]+historys[-1]+history+[speaker1]+historys[-1]+[speaker2]+reply)-1
    history_ids=-len([speaker2]+reply+historys[-1]+[speaker2])-1
    src_ids=-len([speaker2]+reply)-1
    encoder_ids=-len(history+historys[-1]+[speaker2]+[speaker2]+reply)-1
    if len(input_ids)>MAX_INPUTS_LENGTH:
        instance["input_ids"]=instance["input_ids"][-MAX_INPUTS_LENGTH:]
        instance["token_type_ids"]=instance["token_type_ids"][-MAX_INPUTS_LENGTH:]
    instance["allHistory"]=history2
    instance["historyKnl"]=historyKnl
    #instance["mc_token_ids"]=[x if x <=MAX_INPUTS_LENGTH-1 else MAX_INPUTS_LENGTH-1 for x in dataset["mc_token_ids"]]
    instance["allHistory"]=instance["allHistory"] if len(instance["allHistory"]) <=MAX_ALL_HISTORY_LENGTH else instance["allHistory"][-MAX_ALL_HISTORY_LENGTH:]
    instance["historyKnl"]=instance["historyKnl"] if len(instance["historyKnl"]) <=MAX_ALL_KNOWLEDGE_LENGTH else instance["historyKnl"][-MAX_ALL_KNOWLEDGE_LENGTH:]

    if knl_ids<=-MAX_INPUTS_LENGTH:
        print("出现了")
        instance["knl_ids"]=0
    else:
        instance["knl_ids"]=knl_ids
    if history_ids<=-MAX_INPUTS_LENGTH:
        instance["history_ids"]=0
    else:
        instance["history_ids"]=history_ids
    if src_ids<=-MAX_INPUTS_LENGTH:
        instance["src_ids"]=0
    else:
        instance["src_ids"]=src_ids
    if encoder_ids<=-MAX_INPUTS_LENGTH:
        instance["encoder_ids"]=0
    else:
        instance["encoder_ids"]=encoder_ids
    instance["mc_token_ids"]=-1
    #instance["lm_labels"] = [-100]*(len(input_ids)-len(dialog["tgt"]+[eos]))+dialog["tgt"]+[eos]
    #if lm_labels:
    #instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history,historyKnl, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(["<eos>",])
    if current_output is None:
        current_output = []

    for i in range(args.max_length):

        instance = build_input_from_segments(personality, history, historyKnl,current_output, tokenizer, with_eos=False)
        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        knl_ids=torch.tensor(instance["knl_ids"], device=args.device).unsqueeze(0)
        history_ids=torch.tensor(instance["history_ids"], device=args.device).unsqueeze(0)
        src_ids=torch.tensor(instance["src_ids"], device=args.device).unsqueeze(0)
        encoder_ids=torch.tensor(instance["encoder_ids"], device=args.device).unsqueeze(0)
        mc_token_ids=torch.tensor(instance["mc_token_ids"], device=args.device).unsqueeze(0)
        historyKnl1=torch.tensor(instance["historyKnl"], device=args.device).unsqueeze(0)
        allHistory=torch.tensor(instance["allHistory"], device=args.device).unsqueeze(0)

        logits = model(input_ids,
                       knl_ids, history_ids, src_ids,encoder_ids,mc_token_ids,historyKnl1,allHistory,token_type_ids=token_type_ids,tokenizer=tokenizer)

        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)
        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            count=0
            while prev.item() in special_tokens_ids:
                count+=1
                if count >5:
                    break
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")

                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())
    return current_output

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache_movie', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="runs/movie_gpt2_DcDial", help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=50, help="Seed")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.3, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        if args.model == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()
    if args.seed != 0:
        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)
    dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    with open('genetate.txt','w') as file:
        for item in dataset['test']:
            file.write("knowledge：")
            file.write(tokenizer.decode(item["knl"]))
            file.write("\n")
            knl =item["knl"]
            history=item["history"]
            file.write("last utterance:")
            file.write(tokenizer.decode(history[-1]))
            file.write("\n")
            historyKnl=item["historyKnl"]
            with torch.no_grad():
                out_ids= sample_sequence(knl, history, historyKnl,tokenizer, model, args)
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            file.write("generated response:")
            file.write(out_text)
            file.write("\n")
            file.write("golden reply:")
            file.write(tokenizer.decode(item["tgt"]))
            file.write("\n")
        file.write("\n\n")

if __name__ == "__main__":
    run()
