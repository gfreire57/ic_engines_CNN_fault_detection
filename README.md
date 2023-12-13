# TCC: CLASSIFICAÇÃO DE FALHAS EM MOTORES DIESEL A PARTIR DE RESPOSTAS NO DOMÍNIO DO TEMPO POR MEIO DE PROCESSAMENTO DE SINAIS E REDE NEURAL CONVOLUCIONAL
Autor: Gabriel Hasmann
Universidade Federal de Itajubá - 2023


Os dados foram extraídos do site Mendeley: https://data.mendeley.com/datasets/k22zxz29kr/1

O trabalho de referência com explicação dos dados usado como base encontra-se nesse link:
  https://www.mdpi.com/2075-1702/11/5/530

O trabalho desenvolvido pode ser baixado deste repositório.

* Como usar?

  A base de dados utilizada possui dados de vibração torcional de um motor a diesel. Esses dados são simulados (ver trabalho de referência acima). Na base usada, são as informações contidas nos arquivos resposta_vibracao.txt contidos dentro de cada pasta dos 3500 sinais analisados.

  O script "generate_images_and_train_model.ipynb" gera as imagens para cada transformada (STFT - Short Time Fourier Transform - e CWT - Continuous Wavelet Transform). As funções para cada uma estão nos arquvios func_STFT.py e func_WAVELET.py.

  Em seguida, faz o treinamento das redes. Para cada par de caso transformada/ruído, cria-se um objeto RunCNNModel_TCC desenvolvido no arquivo "Classe_treino_CNN_train_Gabriel.py" que por sua vez possui todas as funções básicas necessárias para o treino. O modelo final é salvo na pasta ./Execucoes que será criada caso não exista.
