import pandas as pd
from tqdm import tqdm

def truncar_historico(text, max_caracteres=250):
    """
    Trunca o texto para um número máximo de caracteres.
    
    Parameters:
    - text (str): O texto a ser truncado.
    - max_caracteres (int): Número máximo de caracteres permitidos.
    
    Returns:
    - str: Texto truncado ou original se estiver dentro do limite.
    """
    if pd.isna(text):
        return text  # Retorna NaN ou None sem alterações
    text = str(text)
    if len(text) > max_caracteres:
        return text[:max_caracteres]
    return text

def reduzir_historico_csv(caminho_entrada, caminho_saida, coluna_historico='Historico', max_caracteres=2000, encoding_entrada='utf-8', encoding_saida='utf-8'):
    """
    Lê um CSV, trunca a coluna 'Historico' para um número máximo de caracteres e salva em um novo CSV.
    
    Parameters:
    - caminho_entrada (str): Caminho para o arquivo CSV original.
    - caminho_saida (str): Caminho para salvar o arquivo CSV reduzido.
    - coluna_historico (str): Nome da coluna a ser truncada.
    - max_caracteres (int): Número máximo de caracteres permitidos na coluna 'Historico'.
    - encoding_entrada (str): Encoding do arquivo de entrada.
    - encoding_saida (str): Encoding do arquivo de saída.
    """
    try:
        # Tentar ler o CSV com o encoding especificado
        df = pd.read_csv(caminho_entrada, encoding=encoding_entrada)
    except UnicodeDecodeError:
        # Tentar com outro encoding se houver erro
        try:
            df = pd.read_csv(caminho_entrada, encoding='ISO-8859-1')
            print("Arquivo lido com encoding 'ISO-8859-1'.")
        except UnicodeDecodeError:
            # Tentar com outro encoding se ainda houver erro
            df = pd.read_csv(caminho_entrada, encoding='Windows-1252')
            print("Arquivo lido com encoding 'Windows-1252'.")
    
    # Verificar se a coluna 'Historico' existe
    if coluna_historico not in df.columns:
        raise ValueError(f"A coluna '{coluna_historico}' não foi encontrada no CSV.")
    
    # Aplicar a função de truncamento na coluna 'Historico'
    tqdm.pandas(desc="Truncando 'Historico'")
    df[coluna_historico] = df[coluna_historico].progress_apply(lambda x: truncar_historico(x, max_caracteres))
    
    # Salvar o DataFrame modificado em um novo CSV
    df.to_csv(caminho_saida, index=False, encoding=encoding_saida)
    print(f"Arquivo reduzido salvo em: {caminho_saida}")

if __name__ == "__main__":
    # Definir os caminhos dos arquivos
    caminho_csv_entrada = "2022.csv"  # Substitua pelo caminho do seu arquivo CSV original
    caminho_csv_saida = "2022reduzido.csv"  # Nome do arquivo CSV reduzido
    
    # Chamar a função para reduzir a coluna 'Historico'
    reduzir_historico_csv(
        caminho_entrada=caminho_csv_entrada,
        caminho_saida=caminho_csv_saida,
        coluna_historico='Historico',
        max_caracteres=250,
        encoding_entrada='utf-8',  # Ajuste se necessário
        encoding_saida='utf-8'      # Ajuste se necessário
    )
