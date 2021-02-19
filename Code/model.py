import torch


class Model(torch.nn.Module):
    def __init__(self, seg_model, bert_model):
        super().__init__()
        self.seg_model = seg_model
        self.bert_model = bert_model

    def forward(self, sent, attention, img):
        last_hidden_state = self.bert_model(sent,
                                            attention_mask=attention)[0]
        embedding = last_hidden_state[:, 0, :]
        outputs, _, _ = self.seg_model(img, embedding.squeeze(1))

        outputs = outputs["out"]

        return outputs

    def eval(self):
        self.seg_model.eval()
        self.bert_model.eval()

    def train(self):
        self.seg_model.train()
        self.bert_model.train()
