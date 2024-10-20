
import torch.nn as nn
import torch.optim as optim
from data import *
from config import *
from model.transformer import Transformer
from utils import describe
from data import *

if __name__ == "__main__":

    # 初始化数据集
    dataset = TranslationDataset(source_texts, target_texts, source_vocab, target_vocab)


    # 使用DataLoader创建小批量数据
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


    model = Transformer(len(source_vocab), len(target_vocab), d_model, n_head, n_layers).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 5. 使用DataLoader
    print(source_tokens[0])

    for epoch in range(epochs):
        # print(f"-------------------------Epoch [{epoch + 1} / {epochs}]-------------------------")
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch"):
            enc_inputs, dec_inputs, dec_outputs = batch
            # print("Source Batch:", enc_inputs, "\n", "source batch:", enc_inputs.shape)
            # print("Target Batch:", dec_inputs, "\n", "target_batch:", dec_inputs.shape)
            # print("dec_outputs:", dec_outputs, "\n", "dec_outputs:", dec_outputs.shape)

            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

            # describe(outputs, "outputs")
            # describe(dec_outputs, "dec_outputs")
            loss = criterion(outputs, dec_outputs.view(-1))
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # break
        avg_loss = running_loss / len(dataloader)
        print(f"Loss: {avg_loss:,.4f}")

    torch.save(model, 'model.pth')
    print("模型已保存")