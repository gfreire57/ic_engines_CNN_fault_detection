import os
import glob
import pandas as pd
import numpy as np
from obspy.signal.tf_misfit import cwt
import matplotlib.pyplot as plt
from scipy import signal


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
    print("Pasta com origem dos dados", path)

    return base_dados, pasta_start, pasta_fim, pasta_severidades, path

def add_noise(sig, NOISE_VALUE, FREQUENCY): #, t0
    N = sig.shape[0]
    # print(f"{N =}")
    time = np.linspace(0, N/FREQUENCY, num=262144)
    # signal = signal.values.squeeze()
    # print(f"{time = }")
    # print(f"{signal = }")

    sig_watts = sig ** 2

    # Set a target SNR (Signal to Noise Ratio)
    noisePercentage = NOISE_VALUE/100
    snr = 1/noisePercentage
    target_snr_db = 10 * np.log10(snr)

    # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(sig_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)

    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)

    # Generate an sample of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(sig_watts))
    # Noise up the original signal
    noisySignal = sig + noise
    # df = pd.DataFrame(noisySignal)
    return noisySignal

def generate_images(
        severidade, 
        WITH_NOISE: bool = False,
        NOISE_VALUE: int = 0,
        ):
    
    if NOISE_VALUE == 0:
            WITH_NOISE = False
            
    print(f"Gerando imagens de: {severidade}")
    base_dados, pasta_start, pasta_fim, pasta_severidades, path = selecao_simulacao(severidade)

    # if WITH_NOISE:
    #     plot_name = rf'Plots_STFT_w_noise\noise_{NOISE_VALUE}%'
    # else:
    #     plot_name = rf'Plots_STFT'
    plot_name = rf'NEW_Plots_CWTS_w_noise\noise_{NOISE_VALUE}%'
    SAVE_PATH = os.path.join(os.getcwd(),plot_name, severidade)

    if os.path.isdir(SAVE_PATH) == False:
        os.makedirs(SAVE_PATH)

    # Configuração da simulação    
    # Configuração da simulação   
    fs = 15000  # frequencia de amostragem
    dt = 1/fs
    f_min = 1
    f_max = 2000
    f_corte = 1000
    w0 = 5 #Wavalet cycles
    nf = 1000 #Log division

    # itera por cada ponto analisado
    for ponto in np.arange(start = pasta_start, stop = pasta_fim + 1, step=1):

        IMAGE_PATH = os.path.join(SAVE_PATH, severidade+'_'+str(ponto))
        if os.path.exists(IMAGE_PATH + '.png'):
            print(f"imagem para par {severidade}-{ponto} existe.")
        else:
            # carrega os dados desse ponto
            ponto_path    = os.path.join(path, str(ponto), 'resposta_vibracao.txt')
            sig           = list(np.loadtxt(ponto_path, usecols=1))
            sig           = np.array(sig)

            caminho_para_print = os.path.join("Diesel DataBase", base_dados, '...', str(ponto), 'resposta_vibracao.txt')
            print(caminho_para_print)
        
            # determina intervalo de tempo do teste para plotagem
            n = len(sig)
            t = np.linspace(0, dt * n, n) # tempo do sinal
            
            # Adding noise
            if WITH_NOISE:
                sig = add_noise(
                    sig        = sig,
                    NOISE_VALUE   = NOISE_VALUE,
                    FREQUENCY     = fs
                )

            # Transformada WAVELET
            scalogram = cwt(sig, dt, w0, f_min, f_max, nf=nf)

            # # Filtra pois os fenomenos ocorrem somente em baixas frequencias (experimentação)
            # fmin = 0 # Hz
            # fmax = 1000 # Hz
            # freq_slice = np.where((f >= fmin) & (f <= fmax))
            # f   = f[freq_slice]
            # Sxx = Sxx[freq_slice,:][0]

            # plota
            fig = plt.figure (figsize=(1,1), dpi = 128, clear = True)
            ax = fig.add_subplot(111)
            x, y = np.meshgrid(t,np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))
            ax.pcolormesh(x, y, np.abs(scalogram), cmap='jet', shading = 'auto')
            plt.axis('off')
            plt.margins(0,0)
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,hspace = 0, wspace = 0)
            # IMAGE_PATH = os.path.join(SAVE_PATH, severidade+'_'+str(ponto))
            plt.savefig(IMAGE_PATH)
            plt.close('all')

    print(f"Arquivos salvos na pasta: {SAVE_PATH}")


# severidades = [
#     'normal',
#     'red_pressao',
#     'red_razcomp',
#     'red_comb'
# ]
# WITH_NOISE = True
# # NOISE_VALUE = 75 #%
# NOISES_ANALYSED = [80] # 10, 20, 40. 60, 80
# for noise in NOISES_ANALYSED:
#     for sev in severidades:
#         print(f"--- {noise = } /// severidade: {sev}")
#         generate_images(
#             severidade  = sev,
#             WITH_NOISE  = WITH_NOISE,
#             NOISE_VALUE = noise
#             )