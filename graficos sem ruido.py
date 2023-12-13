import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

'''
PLOTAR SINAIS DA VIBRAÇÃO TORCIONAL CRUS SEM RUIDO
'''

font = {'family' : 'DejaVu Sans',
        # 'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)   

def selecao_simulacao(severidade, dB: str = '0dB', RPM: str = '2500RPM'):
    # Seleção de simulação
    if severidade == 'normal':
        base_dados = 'Base_Severidades_Normais'
        pasta_start = 1
        # qtde_pastas = 250
        pasta_fim = 250
        pasta_severidades = 'normal'

    elif severidade == 'red_pressao':
        base_dados= 'Base_Severidades'
        pasta_start = 1
        pasta_fim = 250
        # qtde_pastas = 250
        pasta_severidades = 'red_pressao'

    elif severidade == 'red_razcomp':
        base_dados= 'Base_Severidades'
        pasta_start = 251
        pasta_fim = 1750
        # qtde_pastas = 1500
        pasta_severidades = 'red_razcomp'

    elif severidade == 'red_comb':
        base_dados= 'Base_Severidades'
        pasta_start = 1751
        pasta_fim = 3250
        # qtde_pastas = 1500
        pasta_severidades = 'red_comb'
    
    else: raise Exception("Houve um problema na selecao de severidade.")
    
    path = os.path.join(os.getcwd(), "Diesel DataBase", base_dados, RPM, dB)
    # print("Pasta com origem dos dados", path)

    return base_dados, pasta_start, pasta_fim, pasta_severidades, path

severidades = {
    'normal'        : 'normal',
    'red_pressao'   : 'redução de pressão no manifold',
    'red_razcomp'   : 'redução da razão de compressão em um cilindro',
    'red_comb'      : 'redução do volume mássico de combustível'
}
WITH_NOISE = True
# NOISE_VALUE = 75 #%
i = 0

fig, axes = plt.subplots(4, 1, figsize=(10, 7), dpi = 100, clear = True) # figsize=(12,4)
for sev in severidades:
    print(f"------ severidade: {sev}")
            
    # print(f"Gerando imagens de: {severidade}")
    base_dados, pasta_start, pasta_fim, pasta_severidades, path = selecao_simulacao(sev)

    # Configuração da simulação    
    fs = 15000  # frequencia de amostragem
    nperseg = 256 # n divisão
    NOISES_ANALYSED = [0, 10, 20, 40, 60, 80]
    colors          = ['-k', '-r', '-m', '-b', '-c', '-y']
    # plt.figure(figsize=(12,4), dpi = 200, clear = True)


    # fig, axes = plt.subplots(4, 1, figsize=(12, 7), dpi = 100, clear = True) # figsize=(12,4)
    # lim_inf = .200
    # lim_sup = .210
    # itera por cada ponto analisado
    # for i, noise in enumerate(NOISES_ANALYSED):
    for ponto in np.arange(start = pasta_start, stop = pasta_fim + 1, step=1):
        # carrega os dados DE UM PONTO SOMENTE
        ponto_path    = os.path.join(path, str(ponto), 'resposta_vibracao.txt')
        print(ponto_path)
        sig           = list(np.loadtxt(ponto_path, usecols=1))
        sig           = np.array(sig)
        time          = list(np.loadtxt(ponto_path, usecols=0))
        time          = np.array(time)

        # Pegar somente intervalo desejado
        # time_ = [t for t in time if lim_inf < t < lim_sup]
        # sig_  = [s for t, s in zip(time, sig) if lim_inf < t < lim_sup]
        
        time_ = np.array(time)
        sig_ = np.array(sig)

        # plota
        # plt.figure (figsize=(1,1), dpi = 128, clear = True)
        axes[i].plot(time_, sig_, colors[0], lw=1, label=sev)
        
        # plt.axis('off')
        axes[i].margins(0,0)
        axes[i].spines['right'].set_visible(True)
        axes[i].spines['top'].set_visible(True)
        axes[i].spines['left'].set_visible(True)
        axes[i].spines['bottom'].set_visible(True)
        # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,hspace = 0, wspace = 0)
        # IMAGE_PATH = os.path.join(SAVE_PATH, severidade+'_'+str(ponto))
        # plt.savefig(IMAGE_PATH)
        # plt.close('all')
        break # RODA SOMENTE UMA IMAGEM PARA VISUALIZAR
        
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=6)
    axes[i].set_title(f"{severidades[sev].title()}")
    axes[i].set_ylabel("Amplitude (Nm)")
    axes[i].set_xlabel("Tempo (s)")
    # axes[i].setylim((-4000, 3500))
    # plt.tight_layout()
    i += 1
fig.tight_layout()
plt.show()
    # break
