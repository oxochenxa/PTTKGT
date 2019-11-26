#Bai 1:
from sys import argv
class Node: 
  
	# Constructor to create a new node 
	def __init__(self, data): 
		self.data = data 
		self.left = None
		self.right = None
		
	def insert(self, data):

		if self.data:
			if data < self.data:
				if self.left is None:
					self.left = Node(data)
				else:
					self.left.insert(data)
			elif data > self.data:
				if self.right is None:
					self.right = Node(data)
				else:
					self.right.insert(data)
		else:
			self.data = data
	def inorderTraversal(self, root):
		res = []
		if root:
			res = self.inorderTraversal(root.left)
			res.append(root.data)
			res = res + self.inorderTraversal(root.right)
		return res
# Compute the "maxDepth" of a tree -- the number of nodes  
# along the longest path from the root node down to the  
# farthest leaf node 
def maxDepth(node): 
	if node is None: 
		return 0 ;  
  
	else : 
  
		# Compute the depth of each subtree 
		lDepth = maxDepth(node.left) 
		rDepth = maxDepth(node.right) 
  
		# Use the larger one 
		if (lDepth > rDepth): 
			return lDepth+1
		else: 
			return rDepth+1
  
if argv[1] == 'height':  
	if __name__ == "__main__":
		f = open(argv[2], "r")
		x = []
		for item in f:
			data = item.split(' ')
			for j in range(len(data)):
				data[j] = int(data[j])
				x.append(data[j])
		#print(x)
		root = Node(x[0])
		for i in range(1,len(x)):
			root.insert(x[i])
		a = root.inorderTraversal(root)
		#print(a)
		print ("Height of tree is %d" %(maxDepth(root)))
		#Write result
		output1 = open(argv[3],"w") 
		output1.write(str(maxDepth(root)))
		output1.close()
#------------------------------------------------------------------------
#Bai 2
from sys import argv
class Node:

	def __init__(self, data):

		self.left = None
		self.right = None
		self.data = data
# Insert Node
	def insert(self, data):

		if self.data:
			if data < self.data:
				if self.left is None:
					self.left = Node(data)
				else:
					self.left.insert(data)
			elif data > self.data:
				if self.right is None:
					self.right = Node(data)
				else:
					self.right.insert(data)
		else:
			self.data = data

# Print the Tree
	def PrintTree(self):
		if self.left:
			self.left.PrintTree()
		print( self.data),
		if self.right:
			self.right.PrintTree()

# Preorder traversal
# Root -> Left ->Right
	def PreorderTraversal(self, root):
		res = []
		if root:
			res.append(root.data)
			res = res + self.PreorderTraversal(root.left)
			res = res + self.PreorderTraversal(root.right)
		return res


if argv[1] == 'preorder':
	if __name__ == "__main__":
		f = open(argv[2], "r")
		x = []
		for item in f:
			data = item.split(' ')
			for j in range(len(data)):
				data[j] = int(data[j])
				x.append(data[j])
		print(x)
		root = Node(x[0])
		for i in range(1,len(x)):
			root.insert(x[i])

		print(root.PreorderTraversal(root))
		#Write result
		output1 = open(argv[3],"w")
		#output1.write(str(root.PreorderTraversal(root)))
		result = root.PreorderTraversal(root)
		for item in result:
			output1.write(str(item)+" ")
		output1.close()

#------------------------------------------------------------------------
#Bai 3
from collections import defaultdict
import math
from sys import argv
import re

def Graph(file):
	temp = None
	for line in file:
		if not re.match("//", line):
			info = line.split(" ")
			if info[0] == temp:
				graph[info[0]][info[1]] = int(info[2].replace('\n', ''))
				if not (info[1] in graph):
					graph[info[1]] = {info[0]: int(info[2].replace('\n', ''))}
			else:
				temp = info[0]
				if not (info[0] in graph):
					graph[info[0]] = {info[1]: int(info[2].replace('\n', ''))}
				else: graph[info[0]][info[1]] = int(info[2].replace('\n', ''))
				if not (info[1] in graph):
					graph[info[1]] = {info[0]: int(info[2].replace('\n', ''))}
				else: graph[info[1]][info[0]] = int(info[2].replace('\n', ''))


def dijkstra(G):
	Dist = defaultdict(lambda: math.inf)
	Prev = defaultdict(None)
	Dist['0'] = 0
	V = list(G.keys())
	Rem = V.copy()
	while len(Rem) > 0:
		u = Rem[0]
		u_value = Dist[u]
		for x in Rem[1:]:
			if Dist[x] < u_value:
				u = x
				u_value = Dist[x]
		Rem.remove(u)
		for v in G[u]:
			if v in Rem:
				Z = min(Dist[v], Dist[u] + G[u][v])
				if Z < Dist[v]:
					Dist[v] = Z
					Prev[v] = u
	return Dist, Prev

if argv[1] == 'dijkstra':
	if __name__ == '__main__':
		graph = {}
		file = open(argv[2], "r")
		Graph(file)
		D, P = dijkstra(graph)
		graph_array = []
		for i in D:
			graph_array.append(i)
		print(graph_array)
		
		for i in range(1, len(graph_array)):
			print(graph_array[i],':',D[graph_array[i]])
			
		result = open(argv[3],"w")
		for i in range(1, len(graph_array)):
			
			result.write(str(graph_array[i])+' '+str(D[graph_array[i]]))
			result.write("\n")
		result.close()
	
	
#------------------------------------------------------------------------
#Bai 4
from sys import argv  
from collections import defaultdict 

#Class to represent a graph 
class Graph: 
  
	def __init__(self,vertices): 
		self.V= vertices #No. of vertices 
		self.graph = [] # default dictionary  
								# to store graph 
		  
   
	# function to add an edge to graph 
	def addEdge(self,u,v,w): 
		self.graph.append([u,v,w]) 
  
	# A utility function to find set of an element i 
	# (uses path compression technique) 
	def find(self, parent, i): 
		if parent[i] == i: 
			return i 
		return self.find(parent, parent[i]) 
  
	# A function that does union of two sets of x and y 
	# (uses union by rank) 
	def union(self, parent, rank, x, y): 
		xroot = self.find(parent, x) 
		yroot = self.find(parent, y) 
  
		# Attach smaller rank tree under root of  
		# high rank tree (Union by Rank) 
		if rank[xroot] < rank[yroot]: 
			parent[xroot] = yroot 
		elif rank[xroot] > rank[yroot]: 
			parent[yroot] = xroot 
  
		# If ranks are same, then make one as root  
		# and increment its rank by one 
		else : 
			parent[yroot] = xroot 
			rank[xroot] += 1
  
	# The main function to construct MST using Kruskal's  
		# algorithm 
	def KruskalMST(self): 
  
		result =[] #This will store the resultant MST 
  
		i = 0 # An index variable, used for sorted edges 
		e = 0 # An index variable, used for result[] 
  
			# Step 1:  Sort all the edges in non-decreasing  
				# order of their 
				# weight.  If we are not allowed to change the  
				# given graph, we can create a copy of graph 
		self.graph =  sorted(self.graph,key=lambda item: item[2]) 
  
		parent = [] ; rank = [] 
  
		# Create V subsets with single elements 
		for node in range(self.V): 
			parent.append(node) 
			rank.append(0) 
	  
		# Number of edges to be taken is equal to V-1 
		while e < self.V -1 : 
			# Step 2: Pick the smallest edge and increment  
					# the index for next iteration 
			u,v,w =  self.graph[i] 
			i = i + 1
			x = self.find(parent, u) 
			y = self.find(parent ,v) 
  
			# If including this edge does't cause cycle,  
						# include it in result and increment the index 
						# of result for next edge 
			if x != y: 
				e = e + 1     
				result.append([u,v,w]) 
				self.union(parent, rank, x, y)             
			# Else discard the edge 
  
		# print the contents of result[] to display the built MST 
		print("Following are the edges in the constructed MST")
		for u,v,weight  in result: 
			#print str(u) + " -- " + str(v) + " == " + str(weight) 
			print ("%d -- %d == %d" % (u,v,weight))
		res = 0
		for i in range(len(result)):
			res = res + result[i][2]
		return res

if argv[1] == 'kruskal':
	if __name__ == '__main__':
		#Read and adjust data
		with open(argv[2]) as f:
			content = f.read().splitlines()
		a = []
		for data in content:
			a.append(data.split(' '))

		print(a)

		for i in range(len(a)):
			for j in range(len(a[i])):
				a[i][j]=int(a[i][j])
		#Number of vertices		
		b = []
		for i in range(len(a)):
			for j in range(0,2):
				if a[i][j] not in b:
					#print(a[i][j])
					b.append(a[i][j])
		print(len(b))
		g = Graph(len(b))
		for i in range(len(a)):
			g.addEdge(int(a[i][0]),int(a[i][1]), int(a[i][2]))
		result = open(argv[3],"w")
		result.write(str(g.KruskalMST()))
		result.close()
	
#------------------------------------------------------------------------
#Bai 5
from collections import defaultdict
import math
from sys import argv
import math
import heapq as hq
import re

def floyd(A, n, s):
	D = [[0 for _ in range(n)] for _ in range(n)] 
	for i in range(n):
		for j in range(n):
			D[i][j] = A[i][j]
	for i in range(n):
		D[i][i] = 0
	for k in range(n):
		for i in range(n):
			for j in range(n):
				if (D[i][k] != infinitive and D[k][j] != infinitive and D[i][k] + D[k][j] < D[i][j]):
					D[i][j] = D[i][k] + D[k][j];
	for i in range(5):
		if(i!=s):
			print(i,D[s][i])
		result = open(argv[3],"w")
		result.write(str(i)+str(D[s][i]))
	result.close()

if argv[1] == 'floyd':
	if __name__ == '__main__':
		infinitive=math.inf

		A = [[0 for _ in range(5)] for _ in range(5)]

		graph = []
		temp = None
		file = open(argv[2], "r")
		for line in file:
			if not re.match("//", line):
				info = line.split(" ")
				if info[0] != info[1]:
					A[int(info[0])][int(info[1])] = int(info[2])
					A[int(info[1])][int(info[0])] = int(info[2])
		for i in range(len(A)):
			for j in range(len(A[i])):
				if i != j and A[i][j] == 0:
					A[i][j] = infinitive

		floyd(A, 5, 0)	




