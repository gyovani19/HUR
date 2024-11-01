import torch
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import re

# Definir o dispositivo
device = 0 if torch.cuda.is_available() else -1

# Inicializar o pipeline de classificação zero-shot com modelo multilíngue
classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    device=device
)

# Definir as categorias com descrições mais detalhadas
labels = {
    "violenta": "violenta, envolvendo intenção de causar dano ou agressão física.",
    "não violenta": "não violenta, ocorrendo sem intenção de causar dano ou agressão física.",
    "indefinido": "indefinido, sem informações suficientes para determinar a natureza da morte."
}

# Definir palavras-chave para cada categoria
palavras_violenta = [
    'assassinato', 'homicídio', 'homicidio', 'atentado', 'ataque',
    'morte violenta', 'violento', 'contundente',
    'crime', 'violência', 'agressão', 'arma de fogo',
    'arma branca', 'golpe', 'agressivo'
]

palavras_nao_violenta = [
    'caída', 'desabamento', 'colisão leve',
    'não violenta', 'nao violenta', 'indefinido', 
    'não homicídio', 'nao homicidio', 'não assassinato', 'nao assassinato',
    'descarga elétrica', 'queda', 'incidente', 'acidente de trânsito',
    'acidente', 'colisão suave', 'lesão leve', 'queda da própria altura'
]

# Função de classificação usando o pipeline com regras complementares
def classify_historico_final(text, classifier, labels, palavras_violenta, palavras_nao_violenta):
    # Classificação zero-shot com hipóteses detalhadas
    candidate_labels = list(labels.keys())
    hypothesis_template = "Esta morte é {}."
    
    result = classifier(
        text, 
        candidate_labels=candidate_labels, 
        hypothesis_template=hypothesis_template
    )
    
    classification = result['labels'][0]
    score = result['scores'][0]
    print(f"Classificação: {classification} (Score: {score:.4f})")
    
    # Se a classificação for 'indefinido', aplicar regras baseadas em palavras-chave
    if classification == "indefinido":
        text_lower = text.lower()
        print("Aplicando regras de palavras-chave.")
        
        # Verificar palavras-chave não violentas primeiro
        for palavra in palavras_nao_violenta:
            if re.search(r'\b' + re.escape(palavra) + r'\b', text_lower):
                print(f"Palavra-chave encontrada (não violenta): {palavra}")
                return "não violenta"
        
        # Verificar palavras-chave violentas
        for palavra in palavras_violenta:
            if re.search(r'\b' + re.escape(palavra) + r'\b', text_lower):
                print(f"Palavra-chave encontrada (violenta): {palavra}")
                return "violenta"
        
        print("Nenhuma palavra-chave encontrada.")
        return "indefinido"
    else:
        return classification

# Caminhos dos arquivos
caminho_csv = "2033reduzido.csv"  # Substitua pelo caminho do seu arquivo CSV
caminho_saida = "2033LLM_zero_shot_com_fallback.csv"

# Carregar o CSV com diferentes encodings
try:
    df = pd.read_csv(caminho_csv, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(caminho_csv, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df = pd.read_csv(caminho_csv, encoding='Windows-1252')

# Verificar se a coluna 'Historico' existe
if 'Historico' not in df.columns:
    raise ValueError("A coluna 'Historico' não foi encontrada no CSV.")

# Inicializar a nova coluna com valores padrão
df['Classificacao_Final'] = 'indefinido'

# Iterar sobre as linhas do DataFrame e classificar cada 'Historico'
for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Classificando históricos"):
    historico = str(row['Historico'])
    classificacao_final = classify_historico_final(
        historico, 
        classifier, 
        labels, 
        palavras_violenta, 
        palavras_nao_violenta
    )
    df.at[idx, 'Classificacao_Final'] = classificacao_final

# Salvar o DataFrame atualizado em um novo CSV
df.to_csv(caminho_saida, index=False, encoding='utf-8')

print(f"Classificação concluída. Arquivo salvo em {caminho_saida}")
