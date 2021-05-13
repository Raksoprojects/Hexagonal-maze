#Oskar Fąk 175984 - Lab6
#zaimplementowałem zmianę labiryntu na hexagonalny
#zaimplementowałem algorytm szukający ścieżki między dwoma
#arbitrarnie wybranymi punktami

import numpy as np
import random as rng
from matplotlib import pyplot as plt
import cv2

def maze(width=81, height=51, complexity=.25, density=.75):
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1]))) #number of components
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))# size of components
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    hex_grid = np.zeros(shape, dtype=bool) # wierzcholki hexagonów, potrzebne później
    #zmienne pomocnicze
    zx = shape[1]//20
    zy = shape[0]//20
    #stworzenie siatki wierzchołków
    for j in range(zx,shape[1]-1,2*zx): #górna część
        hex_grid[0,j] = 1
    #cała reszta siatki, po dwa parzyście i nieparzyście    
    for i in range(zy,shape[0]-1,4*zy):
        
        for j in range(0,shape[1]-1,2*zx):
            hex_grid[i,j] = 1
        for j in range(0,shape[1]-1,2*zx):
            hex_grid[i+zy,j] = 1    
        
        for j in range(zx,shape[1]-1,2*zx):
            hex_grid[i+2*zy,j] = 1
        for j in range(zx,shape[1]-1,2*zx):
            hex_grid[i+3*zy,j] = 1             

    #górna częśc obwódki labiryntu
    for i in range(0,(width//zx)- 1,2):
        Zx1 = np.linspace(i*zx,zx+i*zx,width//20+1, dtype=int) #tworzenie punktów między dwoma wierzchołkami
        Zy1 = np.linspace(zy,0,width//20+1, dtype=int)
        Zy2 = np.linspace(0,zy,width//20+1, dtype=int)
        Z[Zy1,Zx1] = 1 #uzupełnianie tych punktów
        Z[Zy2,Zx1+zx] = 1
    #dolna część obwódki labiryntu        
    for i in range(0,(width//zx)- 1,2):
        Zx1 = np.linspace(i*zx,zx+i*zx,width//20+1, dtype=int) #tworzenie punktów między dwoma wierzchołkami
        Zy1 = np.linspace(shape[0]-zy-1,shape[0]-1,width//20+1, dtype=int)
        Zy2 = np.linspace(shape[0]-1,shape[0]-zy-1,width//20+1, dtype=int)
        Z[Zy1,Zx1] = 1
        Z[Zy2,Zx1+zx] = 1 #uzupełnianie tych punktów
    #prawa i lewa strona obwódki        
    for j in range(0,(height//zy)- 1,4):
        Zy1 = np.linspace(zy+j*zy,zy+(1+j)*zy,width//20+1, dtype=int) # od góry do dołu
        Zy2 = np.linspace(3*zy+j*zy,2*zy+(2+j)*zy,width//20+1, dtype=int)
        Zx1 = np.linspace(0,zx,width//20+1, dtype=int) #lewa do prawej, po lewej
        Zx2 = np.linspace(shape[1]-zx-1,shape[1]-1,width//20+1, dtype=int) # lewa do prawej, po prawej
        Zy3 = np.linspace(zy+(1+j)*zy,zy+j*zy,width//20+1, dtype=int) # od dołu do góry
        Zy4 = np.linspace(2*zy+(2+j)*zy, 3*zy+j*zy, width//20+1, dtype=int)
        Z[Zy1,0] = 1
        Z[Zy1,shape[1]-1] = 1 
        if j + 2 < height//zy-2:
            Z[Zy2,zx] = 1 #pionowe granice po lewej
            Z[Zy2,shape[1]-zx-1] = 1 #pionowe granice po prawej
            Z[Zy1+zy,Zx1] = 1 #ukośne kreski
            Z[Zy1+3*zy,Zx2] = 1
            Z[Zy3+3*zy,Zx1] = 1
            Z[Zy4-zy,Zx2] = 1           
    
    #Make aisles
    for i in range(density):
        #losowanie punktu który jest wierzchołkiem hexagonu
        y,x = np.random.randint(0, shape[1] // 2) * 2, np.random.randint(0, shape[0] // 2) * 2
        while hex_grid[y,x] !=1:
            y,x = np.random.randint(0, shape[1] // 2) * 2, np.random.randint(0, shape[0] // 2) * 2 # pick a random position
        for j in range(complexity):
            neighbours = []
            #sprawdzanie i dodawanie ewentualnych sąsiadów wybranego wierzchołka
            if y> zy and x>zx and hex_grid[y-zy,x-zx] == 1:             
                neighbours.append((y - zy, x - zx))
            if y> zy and x<shape[1]-zx and hex_grid[y-zy,x+zx] == 1:             
                neighbours.append((y - zy, x + zx))
            if y> zy and hex_grid[y-zy,x] == 1 :             
                neighbours.append((y - zy, x))
            if y<shape[0]-zy and x<shape[1] and hex_grid[y+zy,x] == 1:             
                neighbours.append((y + zy, x))    
            if x>zx and hex_grid[y,x-zx] == 1:             
                neighbours.append((y, x - zx))   
            if x<shape[1]-zx and hex_grid[y,x+zx] == 1:             
                neighbours.append((y, x + zx)) 
            if y<shape[0]-zy and x>zx and hex_grid[y+zy,x-zx] == 1 :             
                neighbours.append((y + zy, x - zx))
            if y<shape[0]-zy and x<shape[1]-zx and hex_grid[y+zy,x+zx] == 1:             
                neighbours.append((y + zy, x + zx))                
            if len(neighbours): #jeżeli są jacyś sąsiedzi
                #wybieramy losowego sąsiada
                y_,x_ = neighbours[np.random.randint(0, len(neighbours))]
                zy_ = np.linspace(y,y_,width//20+1,dtype=int)
                zx_ = np.linspace(x,x_,width//20+1,dtype=int) #uzupełniamy punkty między nimi
                Z[zy_,zx_] = 1 #stowrzenie ściany
                hex_grid[y_,x_] = 0 # usunięcie tych wierzchołków z grida
                hex_grid[y,x] = 0
                x, y = x_, y_  
    return Z

#klasa punktów na ścieżce
class Vertex:
    def __init__(self,x_coord,y_coord):
        self.x=x_coord
        self.y=y_coord
        self.d=float('inf') #distance from source
        self.parent_x=None
        self.parent_y=None
        self.processed=False
        self.index_in_queue=None

#Pobierz sąsiadów
def get_neighbors(mat,r,c):
    shape=mat.shape
    neighbors=[]
    #sprawdzamy czy nie wychodzimy za obszar
    if r > 0 and not mat[r-1][c].processed:
         neighbors.append(mat[r-1][c])
    if r < shape[0] - 1 and not mat[r+1][c].processed:
            neighbors.append(mat[r+1][c])
    if c > 0 and not mat[r][c-1].processed:
        neighbors.append(mat[r][c-1])
    if c < shape[1] - 1 and not mat[r][c+1].processed:
            neighbors.append(mat[r][c+1])
    return neighbors

def bubble_up(queue, index):
    
    if index <= 0:
        return queue
    p_index=(index-1)//2
    if queue[index].d < queue[p_index].d:
            queue[index], queue[p_index]=queue[p_index], queue[index]
            queue[index].index_in_queue=index
            queue[p_index].index_in_queue=p_index
            queue = bubble_up(queue, p_index)
    return queue
    
def bubble_down(queue, index):
    length=len(queue)
    lc_index=2*index+1
    rc_index=lc_index+1
    if lc_index >= length:
        return queue
    if lc_index < length and rc_index >= length: 
        if queue[index].d > queue[lc_index].d:
            queue[index], queue[lc_index]=queue[lc_index], queue[index]
            queue[index].index_in_queue=index
            queue[lc_index].index_in_queue=lc_index
            queue = bubble_down(queue, lc_index)
    else:
        small = lc_index
        if queue[lc_index].d > queue[rc_index].d:
            small = rc_index
        if queue[small].d < queue[index].d:
            queue[index],queue[small]=queue[small],queue[index]
            queue[index].index_in_queue=index
            queue[small].index_in_queue=small
            queue = bubble_down(queue, small)
    return queue

def get_distance(img,u,v):
    return 0.1 + (float(img[v][0])-float(img[u][0]))**2+(float(img[v][1])-float(img[u][1]))**2+(float(img[v][2])-float(img[u][2]))**2

def draw_path(img,path, thickness=2):
    #funkcja rysująca ścieżkę
    x0,y0=path[0]
    for vertex in path[1:]:
        x1,y1=vertex
        cv2.line(img,(x0,y0),(x1,y1),(255,0,0),thickness)
        x0,y0=vertex
#główna funckja szukająca najkrótszej ścieżki
def find_shortest_path(img,src,dst):
    pq=[] #stworzenie kolejki priorytetowej
    source_x=src[0] #punkt startowy
    source_y=src[1]
    dest_x=dst[0] #punkt końcowy
    dest_y=dst[1]
    img_rows,img_col=img.shape[0],img.shape[1] #rozmiary obrazka
    matrix = np.full((img_rows, img_col), None) 
    for r in range(img_rows):
        for c in range(img_col):
            matrix[r][c]=Vertex(c,r) #utworzenie macierzy klas
            matrix[r][c].index_in_queue=len(pq)
            pq.append(matrix[r][c]) # dodanie punktu do kolejki
    matrix[source_y][source_x].d=0 #ustawienie odległości początkowej
    pq=bubble_up(pq, matrix[source_y][source_x].index_in_queue)#dodanie do kolejki
    while len(pq) > 0:
        u=pq[0]
        u.processed=True
        pq[0]=pq[-1]
        pq[0].index_in_queue=0
        pq.pop()
        pq=bubble_down(pq,0)
        neighbors = get_neighbors(matrix,u.y,u.x) # pobranie sąsiadów
        for v in neighbors:
            dist=get_distance(img,(u.y,u.x),(v.y,v.x)) #odległość pierwszego punktu w kolejce od sąsiada
            if u.d + dist < v.d:
                v.d = u.d+dist
                v.parent_x=u.x
                v.parent_y=u.y
                idx=v.index_in_queue
                pq=bubble_down(pq,idx)
                pq=bubble_up(pq,idx)
    #zapisanie ścieżki do celu                      
    path=[]
    iter_v=matrix[dest_y][dest_x]
    path.append((dest_x,dest_y))
    while(iter_v.y!=source_y or iter_v.x!=source_x):
        path.append((iter_v.x,iter_v.y))
        iter_v=matrix[iter_v.parent_y][iter_v.parent_x]

    path.append((source_x,source_y))
    return path

if __name__ == "__main__":

    #utworzenie labiryntu
    Z = maze(80,80)
    Z[12*Z.shape[0]//80:16*Z.shape[0]//80,76*Z.shape[1]//80] = 0
    plt.figure(figsize=(10, 5))
    plt.imshow(Z, cmap=plt.cm.binary, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    plt.savefig('maze1.png') #zapisanie labiryntu do obrazka

    #wyrysowanie ścieżki
    img = cv2.imread('maze1.png') # zczytanie zapisanego obrazka labiryntu
    p = find_shortest_path(img, (362,350), (683,123))
    draw_path(img,p)
    plt.figure(figsize=(10,5))
    plt.imshow(img)
    plt.show()
    