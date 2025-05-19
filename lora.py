import torch
import peft

class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.project_in = torch.nn.Linear(10,5)

    def forward(self, x):
        return self.project_in(x)
                
base = Base()
print('\nFull:', sum(p.numel() for p in base.parameters()))
print('\n',base,'\n')


class Lora(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.project_in = torch.nn.Linear(10,5)
        self.lora = torch.nn.Sequential(
            torch.nn.Linear(10,2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(2,5, bias=False)
        )

    def forward(self, x):
        x = self.project_in(x)
        return x + self.lora(x)
    
lora = Lora()
print('=='*50, '\n') #print the parameters of 
print('project_in', sum(p.numel() for p in lora.project_in.parameters()))
print('lora', sum(p.numel() for p in lora.lora.parameters()))
print('Full:', sum(p.numel() for p in lora.parameters()))
print('\n',lora,'\n')


conf = peft.LoraConfig(
    r=2,
    lora_alpha=4,
    lora_dropout=0.1,
    target_modules=["project_in"]
)

boom = peft.get_peft_model(Base, conf)
print('=='*50, '\n')
print('Full:', sum(p.numel() for p in boom.parameters()))
print('\n',boom,'\n')



