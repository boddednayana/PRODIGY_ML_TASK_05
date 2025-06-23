import os, cv2, torch
import numpy as np
from torchvision import models, transforms
from torch import nn
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.metrics import accuracy_score

DATA_DIR = r"C:\Users\konat\OneDrive\PRODIGY_ML_TASK_05\archive (5)\food-101\food-101\images"
TRAIN_FILE = r"C:\Users\konat\OneDrive\PRODIGY_ML_TASK_05\archive (5)\food-101\food-101\meta\train.txt"
TEST_FILE = r"C:\Users\konat\OneDrive\PRODIGY_ML_TASK_05\archive (5)\food-101\food-101\meta\test.txt"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),       
    transforms.RandomRotation(degrees=10),        
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
    transforms.ColorJitter(brightness=0.1, contrast=0.1), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def limited_class_paths(txt_file, limit=100):
    with open(txt_file, 'r') as f:
        lines = f.read().splitlines()
    class_count = defaultdict(int)
    limited_paths = []
    for path in lines:
        cls = path.split('/')[0]
        if class_count[cls] < limit:
            limited_paths.append(path)
            class_count[cls] += 1
    return limited_paths

def load_images(paths, base_dir):
    X, y = [], []
    for rel_path in paths:
        label = rel_path.split('/')[0]
        img_path = os.path.join(base_dir, rel_path + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = transform(img).numpy()
        X.append(img)
        y.append(label)
    return np.array(X, dtype=np.float16), np.array(y)

train_paths = limited_class_paths(TRAIN_FILE, limit=50)
test_paths = limited_class_paths(TEST_FILE, limit=10)

X_train, y_train = load_images(train_paths, DATA_DIR)
X_test, y_test = load_images(test_paths, DATA_DIR)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_enc, dtype=torch.long)

train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_data = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(le.classes_))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(10):  
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb)
        pred_labels = preds.argmax(dim=1).cpu().numpy()
        y_true.extend(yb.numpy())
        y_pred.extend(pred_labels)
 
acc = accuracy_score(y_true, y_pred)
print(f"\n Test Accuracy: {acc * 100:.2f}%")

calorie_map = {
    "apple_pie": 296,
    "baby_back_ribs": 320,
    "baklava": 334,
    "beef_carpaccio": 180,
    "beef_tartare": 195,
    "beet_salad": 130,
    "beignets": 289,
    "bibimbap": 560,
    "bread_pudding": 250,
    "breakfast_burrito": 305,
    "bruschetta": 170,
    "caesar_salad": 180,
    "cannoli": 225,
    "caprese_salad": 210,
    "carrot_cake": 320,
    "ceviche": 160,
    "cheesecake": 321,
    "cheese_plate": 300,
    "chicken_curry": 240,
    "chicken_quesadilla": 320,
    "chicken_wings": 430,
    "chocolate_cake": 370,
    "chocolate_mousse": 250,
    "churros": 280,
    "clam_chowder": 180,
    "club_sandwich": 320,
    "crab_cakes": 310,
    "creme_brulee": 210,
    "croque_madame": 390,
    "cup_cakes": 240,
    "deviled_eggs": 200,
    "donuts": 452,
    "dumplings": 260,
    "edamame": 120,
    "eggs_benedict": 290,
    "escargots": 200,
    "falafel": 333,
    "filet_mignon": 300,
    "fish_and_chips": 340,
    "foie_gras": 400,
    "french_fries": 312,
    "french_onion_soup": 180,
    "french_toast": 310,
    "fried_calamari": 280,
    "fried_rice": 228,
    "frozen_yogurt": 130,
    "garlic_bread": 206,
    "gnocchi": 265,
    "greek_salad": 220,
    "grilled_cheese_sandwich": 320,
    "grilled_salmon": 233,
    "guacamole": 150,
    "gyoza": 210,
    "hamburger": 295,
    "hot_and_sour_soup": 150,
    "hot_dog": 290,
    "huevos_rancheros": 230,
    "hummus": 160,
    "ice_cream": 207,
    "lasagna": 310,
    "lobster_bisque": 240,
    "lobster_roll_sandwich": 280,
    "macaroni_and_cheese": 320,
    "macarons": 150,
    "miso_soup": 90,
    "mussels": 172,
    "nachos": 340,
    "omelette": 154,
    "onion_rings": 275,
    "oysters": 170,
    "pad_thai": 320,
    "paella": 300,
    "pancakes": 227,
    "panna_cotta": 210,
    "peking_duck": 340,
    "pho": 240,
    "pizza": 285,
    "pork_chop": 250,
    "poutine": 320,
    "prime_rib": 310,
    "pulled_pork_sandwich": 325,
    "ramen": 280,
    "ravioli": 260,
    "red_velvet_cake": 320,
    "risotto": 300,
    "samosa": 260,
    "sashimi": 130,
    "scallops": 140,
    "seaweed_salad": 90,
    "shrimp_and_grits": 370,
    "spaghetti_bolognese": 310,
    "spaghetti_carbonara": 330,
    "spring_rolls": 140,
    "steak": 271,
    "strawberry_shortcake": 280,
    "sushi": 200,
    "tacos": 280,
    "takoyaki": 220,
    "tiramisu": 240,
    "tuna_tartare": 180,
    "waffles": 291,
    "walnut_brownie": 320,
    "weird_dish": 250  # fallback
}

def predict_calories(img_tensor):
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)
        out = model(img_tensor)
        pred_idx = out.argmax(1).item()
        label = le.inverse_transform([pred_idx])[0]
        calories = calorie_map.get(label, "N/A")
        return label, calories

# === Example prediction ===
sample_img = X_test_tensor[0]
label, cal = predict_calories(sample_img)
print(f"\nPrediction: {label} | Estimated Calories: {cal} kcal")