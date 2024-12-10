import numpy as np

'''
적정 범위
pH(RVI): 5.5 ~ 6.2
P(GNDVI): 250 ~ 350
OM(MTVI1): 20 ~ 30
K(RVI): 0.5 ~ 0.6
Ca(GRVI): 4.5 ~ 5.5
Mg(RVI): 1.5 ~ 2.0
EC(RVI): 0.0 ~ 2.0
'''

def main():
    pH_list = []
    P_list = []
    OM_list = []
    K_list = []
    Ca_list = []
    Mg_list = []
    EC_list = []

    for i in np.arange(0, 1.1, 0.01):
        pH = 2.11*i + 7.572
        P = 554.976*i - 264.133
        OM = 90.083*i - 32.187
        K = 5.581*i - 2.545
        Ca = 6.045*i + 3.977
        Mg = 4.120*i + 0.943
        EC = 1.489*i + 0.151

        if 5.5 <= pH <= 6.2:
            pH_list.append(round(pH, 2))
        if 250 <= P <= 350:
            P_list.append(round(P, 2))
        if 20 <= OM <= 30:
            OM_list.append(round(OM, 2))
        if 0.5 <= K <= 0.6:
            K_list.append(round(K, 2))
        if 4.5 <= Ca <= 5.5:
            Ca_list.append(round(Ca, 2))
        if 1.5 <= Mg <= 2.0:
            Mg_list.append(round(Mg, 2))
        if 0.0 <= EC <= 2.0:
            EC_list.append(round(EC, 2))

    print("pH values:", pH_list)
    print("P values:", P_list)
    print("OM values:", OM_list)
    print("K values:", K_list)
    print("Ca values:", Ca_list)
    print("Mg values:", Mg_list)
    print("EC values:", EC_list)


if __name__ == '__main__':
    main()
