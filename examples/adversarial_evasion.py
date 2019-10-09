from rfml.attack import fgsm
from rfml.data import build_dataset
from rfml.nn.eval import compute_accuracy
from rfml.nn.model import build_model

from torch.utils.data import DataLoader

_, _, test, le = build_dataset(dataset_name="RML2016.10a", test_pct=0.9)
mask = test.df["SNR"] >= 18
model = build_model(model_name="CNN", input_samples=128, n_classes=len(le))
model.load("cnn.pt")

acc = compute_accuracy(model=model, data=test, le=le, mask=mask)
print("Normal (no attack) Accuracy on Dataset: {:.3f}".format(acc))

spr = 10  # dB
right = 0
total = 0
dl = DataLoader(test.as_torch(le=le, mask=mask), shuffle=True, batch_size=512)
for x, y in dl:
    adv_x = fgsm(x, y, spr=spr, input_size=128, sps=8, net=model)

    predictions = model.predict(adv_x)
    right += (predictions == y).sum().item()
    total += len(y)

adv_acc = float(right) / total
print("Adversarial Accuracy with SPR of {} dB attack: {:.3f}".format(spr, adv_acc))
print("FGSM Degraded Model Accuracy by {:.3f}".format(acc - adv_acc))
