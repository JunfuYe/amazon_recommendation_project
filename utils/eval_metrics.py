import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import random

def export_vector_results_dynamic_stochastic_lite():
    base_path = '/content/drive/My Drive/test/'

    from google.colab import drive
    drive.mount('/content/drive')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Table 2 and Feature Matrix
    df_map = pd.read_csv(base_path + 'item_feature_index.csv')
    if 'row_idx' not in df_map.columns and 'Unnamed: 0' in df_map.columns:
        df_map.rename(columns={'Unnamed: 0': 'row_idx'}, inplace=True)
    features_array = np.load(base_path + 'item_features_resnet50.npy', allow_pickle=True)

    candidate_asins, candidate_vectors = [], []
    for _, row in df_map.iterrows():
        r_idx = int(row.get('row_idx', 0))
        if r_idx < features_array.shape[0]:
            candidate_asins.append(str(row['item_id']))
            candidate_vectors.append(torch.tensor(features_array[r_idx]).float())

    candidate_matrix = torch.stack(candidate_vectors).to(device)
    asin_to_idx = {asin: i for i, asin in enumerate(candidate_asins)}

    try: df2 = pd.read_csv(base_path + 'table2.csv')
    except: df2 = pd.read_csv(base_path + 'table2.csv', encoding='ISO-8859-1')

    # 2. loading model
    model = VectorRetrievalLSTM(visual_dim=2048).to(device)
    model.load_state_dict(torch.load(base_path + 'vector_model.pth', map_location=device))
    model.eval()

    # 3. loading test user data
    try: df1 = pd.read_csv(base_path + 'table1.csv')
    except: df1 = pd.read_csv(base_path + 'table1.csv', encoding='ISO-8859-1')
    df1.rename(columns={'user_id': 'reviewerID', 'parent_asin': 'asin', 'timestamp': 'unixReviewTime'}, inplace=True)
    if 'unixReviewTime' in df1.columns: df1.sort_values(['reviewerID', 'unixReviewTime'], inplace=True)

    grouped = df1.groupby('reviewerID')['asin'].apply(list).reset_index()
    valid_groups = grouped[grouped['asin'].map(len) >= 3]

    # 4. Batch prediction and export
    results = []
    sample_users = valid_groups.sample(n=min(500, len(valid_groups)), random_state=42)

    with torch.no_grad():
        for _, row in sample_users.iterrows():
            user_id = row['reviewerID']
            original_history = [str(a) for a in row['asin'][:-1]]

            current_history = original_history.copy()
            taboo_set = set(current_history)

            # Add probability variable r
            r = random.random()
            force_skip_first = (r > 0.5)

            max_retries = 10
            final_rec_asin = None
            final_title = ""

            for attempt in range(max_retries):
                input_asins = current_history[-5:]
                visual_seq = []
                for asin in input_asins:
                    if asin in asin_to_idx:
                        visual_seq.append(candidate_vectors[asin_to_idx[asin]])
                    else:
                        visual_seq.append(torch.zeros(2048))

                pad_len = 5 - len(input_asins)
                for _ in range(pad_len): visual_seq.insert(0, torch.zeros(2048))

                input_tensor = torch.stack(visual_seq).unsqueeze(0).to(device)

                # model prediction
                pred_vector = model(input_tensor)

                # Calculate similarity and add it to the blacklist for blocking
                similarities = F.cosine_similarity(pred_vector, candidate_matrix).view(-1)
                for taboo_asin in taboo_set:
                    if taboo_asin in asin_to_idx:
                        similarities[asin_to_idx[taboo_asin]] = -float('inf')

                top1_idx = torch.argmax(similarities).item()
                rec_asin = candidate_asins[top1_idx]

                r_row = df2[df2['asin'] == rec_asin]

                # Get description for backend judgment
                raw_desc = r_row.iloc[0].get('description', '') if not r_row.empty else ''
                raw_title = str(r_row.iloc[0]['title']) if not r_row.empty else 'unknown title'

                # Check if the description is empty
                is_empty_desc = pd.isna(raw_desc) or str(raw_desc).strip() == '' or str(raw_desc).lower() == 'nan'

                # Force skip logic: If it is the first loop and r>0.5, force it to be treated as an empty description
                if attempt == 0 and force_skip_first:
                    is_empty_desc = True

                if is_empty_desc:
                    # Pretend to make a purchase, add it to history, and then try again!
                    current_history.append(rec_asin)
                    taboo_set.add(rec_asin)
                    continue
                else:
                    # Found the perfect and compliant product! Exit loop
                    final_rec_asin = rec_asin
                    final_title = raw_title
                    break

            if final_rec_asin is None:
                final_rec_asin = rec_asin
                final_title = "Reached maximum retry count"

            results.append({
                "user_id": user_id,
                "history_asin_list": ", ".join(current_history),
                "pred_asin": final_rec_asin,
                "pred_title": final_title
            })

    df_export = pd.DataFrame(results)
    export_path = base_path + 'vector_llm_tasks_final_lite.csv'
    df_export.to_csv(export_path, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    export_vector_results_dynamic_stochastic_lite()