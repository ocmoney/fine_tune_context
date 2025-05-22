import torch
import transformers


tkz = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
plc = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
ref = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

ref.eval()
for p in ref.parameters():p.requires_grad_(False)


optm=torch.optim.AdamW(plc.parameters(), lr=1e-5)
beta=0.1

qry = "question: what is the capital of france? answer:"
pos = "paris"
neg = "london"




def tokenise(qry,res):

    qry_ids = tkz(qry,return_tensors="pt", add_special_tokens=False).input_ids
    res_ids = tkz(res,return_tensors="pt", add_special_tokens=False).input_ids
    acc_ids = torch.cat((qry_ids,res_ids),dim=1)
    atn_msk = torch.ones_like(acc_ids)
    lbl_ids = acc_ids.clone()
    lbl_ids[:, :qry_ids.size(-1)] = -100
    return acc_ids,atn_msk,lbl_ids

def sum_log_probs(model, ids, msk, lbl):
    out = model(input_ids=ids, attention_mask=msk)
    log = out.logits.log_softmax(-1)[:, :-1]
    tgt = lbl[:, 1:].masked_fill(lbl[:, 1:] == -100, 0).unsqueeze(-1)
    tok = log.gather(2, tgt).squeeze(-1)
    msk = lbl[:, 1:] != -100
    return tok[msk].sum(-1)


ids_pos, atn_msk_pos, lbl_pos = tokenise(qry, pos)
ids_neg, atn_msk_neg, lbl_neg = tokenise(qry, neg)


with torch.no_grad():
    log_ref_pos = sum_log_probs(ref, ids_pos, atn_msk_pos, lbl_pos)
    log_ref_neg = sum_log_probs(ref, ids_neg, atn_msk_neg, lbl_neg)


for step in range(2):
    log_plc_pos = sum_log_probs(plc, ids_pos, atn_msk_pos, lbl_pos)
    log_plc_neg = sum_log_probs(plc, ids_neg, atn_msk_neg, lbl_neg)

    delta_pos=log_plc_pos - log_ref_pos
    delta_neg=log_plc_neg - log_ref_neg

    mrgs = delta_pos - delta_neg
    loss = -torch.log(torch.sigmoid(beta*mrgs))

    optm.zero_grad()
    loss.backward()
    optm.step()

    print(f'{loss.item():.4f}')