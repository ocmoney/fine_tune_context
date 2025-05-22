import torch
import transformers

def setup_models(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Setup tokenizer and models"""
    tkz = transformers.AutoTokenizer.from_pretrained(model_name)
    plc = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    ref = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    
    # Freeze reference model
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    
    return tkz, plc, ref

def tokenise(tkz, qry, res):
    """Tokenize query and response"""
    qry_ids = tkz(qry, return_tensors="pt", add_special_tokens=False).input_ids
    res_ids = tkz(res, return_tensors="pt", add_special_tokens=False).input_ids
    acc_ids = torch.cat((qry_ids, res_ids), dim=1)
    atn_msk = torch.ones_like(acc_ids)
    lbl_ids = acc_ids.clone()
    lbl_ids[:, :qry_ids.size(-1)] = -100
    return acc_ids, atn_msk, lbl_ids

def sum_log_probs(model, ids, msk, lbl):
    """Calculate sum of log probabilities"""
    out = model(input_ids=ids, attention_mask=msk)
    log = out.logits.log_softmax(-1)[:, :-1]
    tgt = lbl[:, 1:].masked_fill(lbl[:, 1:] == -100, 0).unsqueeze(-1)
    tok = log.gather(2, tgt).squeeze(-1)
    msk = lbl[:, 1:] != -100
    return tok[msk].sum(-1)

def train_step(plc, ref, optm, ids_pos, atn_msk_pos, lbl_pos, ids_neg, atn_msk_neg, lbl_neg, beta=0.1):
    """Perform one training step"""
    # Get reference log probs
    with torch.no_grad():
        log_ref_pos = sum_log_probs(ref, ids_pos, atn_msk_pos, lbl_pos)
        log_ref_neg = sum_log_probs(ref, ids_neg, atn_msk_neg, lbl_neg)
    
    # Get policy log probs
    log_plc_pos = sum_log_probs(plc, ids_pos, atn_msk_pos, lbl_pos)
    log_plc_neg = sum_log_probs(plc, ids_neg, atn_msk_neg, lbl_neg)
    
    # Calculate deltas
    delta_pos = log_plc_pos - log_ref_pos
    delta_neg = log_plc_neg - log_ref_neg
    
    # Calculate loss
    mrgs = delta_pos - delta_neg
    loss = -torch.log(torch.sigmoid(beta * mrgs))
    
    # Optimize
    optm.zero_grad()
    loss.backward()
    optm.step()
    
    return loss.item()

# Example usage
if __name__ == "__main__":
    # Setup
    tkz, plc, ref = setup_models()
    optm = torch.optim.AdamW(plc.parameters(), lr=1e-5)
    
    # Example data
    qry = "question: what is the capital of france? answer:"
    pos = "paris"
    neg = "london"
    
    # Tokenize
    ids_pos, atn_msk_pos, lbl_pos = tokenise(tkz, qry, pos)
    ids_neg, atn_msk_neg, lbl_neg = tokenise(tkz, qry, neg)
    
    # Train
    for step in range(2):
        loss = train_step(plc, ref, optm, ids_pos, atn_msk_pos, lbl_pos, ids_neg, atn_msk_neg, lbl_neg)
        print(f'{loss:.4f}')