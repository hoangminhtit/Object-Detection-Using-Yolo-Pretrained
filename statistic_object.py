import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_json('./data/object_count.json', typ='series').to_frame(name='count').reset_index()
df.columns = ['object', 'count']

# Vẽ biểu đồ
plt.figure(figsize=(6, 4))
plt.bar(x=df['object'], height=df['count'], color='skyblue')
plt.xlabel('Object')
plt.ylabel('Count')
plt.title('Object Count per Frame')
plt.tight_layout()

# Lưu biểu đồ
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # tạo chuỗi thời gian hợp lệ
plt.savefig(f'./plots/object_count_{timestamp}.png')
print(f"Saved plot: ./plots/object_count_{timestamp}.png")