import matplotlib.pyplot as plt
from file import COUNT

def plot(file_name):

    hdl = open(file_name)
    data = []
    for line in hdl:

        try:

            tmp = float(line.rstrip('\n').split(',')[1])
            data.append(tmp)
        except ValueError:
            pass

    #del data[0]

    data = [float(n) for n in data]
    data = sorted(data)
    hdl.close()

    plt.plot(data, 'ro', markersize=1)
    plt.axhline(y=0.5)
    plt.axvline(x=1000)
    plt.savefig("./image/test" + str(COUNT) + ".png")
    plt.show()


def plotFromList(data,lable):
    pairs = [(data[i,0],lable[i,0])for i in range(len(data))]
    #pairs = [(data[i],lable[i,0])for i in range(len(data))]
    pairs = sorted(pairs,key=lambda x:x[0])
    plt.plot([x[0] for x in pairs],'bx', markersize=1)
    #plt.plot([x[1] for x in pairs],'ro', markersize=1)
    plt.savefig("./image/train" + str(COUNT) + ".png")
    plt.show()


