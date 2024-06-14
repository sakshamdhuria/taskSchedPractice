#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <functional>
#include <random>
#include <unordered_set>
#include <map>
#include <set>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unistd.h>



//each node has four elements - weight, pathWeight, level, children
struct Node {
    int weight;
    int level;
    std::atomic<bool> markedBit = false; //remove ID
    long long pathWeight = LLONG_MAX;
    std::vector<Node*> children;
    std::vector<Node*> parents;

    Node(int weight, int level) : weight(weight), level(level) {}
};

//n is the graph size that is desired
//randomly generates values from 0 to n-1 for every node and starts with level 0
//sets the 0 (starting node) pathWeight as 0
//generates a unique set of parents from the nodes that have already been considered
//for each index it connects it and increments the level of the child 
std::vector<Node*> generateAcyclicGraph(int n, unsigned int seed) {
    //n = number of nodes

    std::vector<Node*> nodes;
    nodes.reserve(n);

    //random objects
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dis(0, n-1);

    // Create nodes
    for (int i = 0; i < n; ++i) {
        nodes.push_back(new Node(dis(gen), 0));  // Level will be updated later
    }

    if (!nodes.empty()) {
        nodes[0]->pathWeight = 0;
    }

    // Create an acyclic graph
    for (int i = 1; i < n; ++i) {
        // Randomly choose a set of unique parent from the existing nodes
        std::set<int> parentIndices;
        std::uniform_int_distribution<int> levelDist(1, std::min(i,14));
        int levelOfMyNode = levelDist(gen);

        std::uniform_int_distribution<int> parentDist(0, levelOfMyNode-1); 
        std::uniform_int_distribution<int> parentChoosingDist(0, i-1);
        int howManyParents = parentDist(gen)+1; 
        //std::cout<<i<<" " <<levelOfMyNode<< " "<< howManyParents<< std::endl;
        while(((int) parentIndices.size()) < howManyParents){
          int indCons = parentChoosingDist(gen);
          if(nodes[indCons]->level<levelOfMyNode){
            parentIndices.insert(indCons);
          }
        }

        //for each index in the set just creating that graph 
        int maxParentLevel = 0;
        for (int parentIndex : parentIndices) {
            maxParentLevel = std::max(nodes[parentIndex]->level, maxParentLevel);
            nodes[i]->pathWeight = std::min(nodes[i]->pathWeight, nodes[parentIndex]->pathWeight + nodes[i]->weight);
            nodes[parentIndex]->children.push_back(nodes[i]);
            nodes[i]->parents.push_back(nodes[parentIndex]);
        }
        nodes[i]->level = maxParentLevel + 1;
    }

    return nodes;
}

//n is the seed set size
//generates <=n nodes of the graph then reassign there value with a value from 0 to nodeSize - 1 like above
std::vector<Node*> generateSeedSet(int n, int nodeSize, std::vector<Node*> &graph, unsigned int seed){
  //n = seed set size
  //nodeSize = node size of the graph
  //want to return a vector of Node that are the nodes which have been changed to a different random weight

  std::vector<Node*> seedSet;
  seedSet.reserve(n);

  //random objects
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dis(0, nodeSize-1);
  std::set<int> indices;//optimizie gen with hash map
  while((int)indices.size()<n){
    int indToChange = dis(gen);
    indices.insert(indToChange);
  }

  for(int ind: indices){
    Node* changingNode = graph[ind];
    int newWeight = dis(gen);
    changingNode->weight = newWeight;
    seedSet.push_back(changingNode);
  }

  return seedSet;
}

void printGraph(Node* root) {
    if (!root) return;

    std::map<Node*, int> nodeIds;  // Map nodes to unique IDs
    int nodeId = 0;  // Start assigning IDs from 0

    std::queue<Node*> q;
    q.push(root);

    while (!q.empty()) {
        Node* current = q.front();
        q.pop();

        if (nodeIds.find(current) == nodeIds.end()) {
            nodeIds[current] = nodeId++;
        }

        std::cout << "Node ID: " << nodeIds[current] << ", Level: " << current->level << ", Children: ";
        for (Node* child : current->children) {
            if (nodeIds.find(child) == nodeIds.end()) {
                nodeIds[child] = nodeId++;
                q.push(child);
            }
            std::cout << nodeIds[child] << " ";
        }
        std::cout << std::endl;
    }
}

//computes a node
//returns true if new path weight is different from old path weight 
bool calcNode(Node* nd){
  long long oldPathWeight = nd->pathWeight;
  long long newPathWeight = LLONG_MAX;
  for(Node* parent :  nd->parents){
    newPathWeight= std::min(parent->pathWeight, newPathWeight);
  }
  newPathWeight+= nd->weight;
  nd->markedBit = false;
  if(newPathWeight==oldPathWeight){
    return false;
  }else{
    nd->pathWeight = newPathWeight;
    return true;
  }
}
 
//bfs using seedSet 
void sequentialBFS(std::vector<Node*> &graph, std::vector<Node*> seedSet, int maxLevel){
  std::vector<std::queue<Node*>> vq(maxLevel+1);

  for(Node* nd : seedSet){
    if(!nd->markedBit){
      vq[nd->level].push(nd);
      nd->markedBit = true;
    }
  }
  for(int level = 0; level<=maxLevel; level++){
    std::queue<Node*>& q = vq[level];
    if(vq[level].size()==0){
      continue;
    }
    while(!q.empty()){
      Node* current = q.front();
      q.pop();
      usleep(10);
      if(calcNode(current)){
        for(Node* child : current->children){
          if(!child->markedBit){
            vq[child->level].push(child);
            child->markedBit = true;
          }
        }
      }
    }
  }
}

void testingSequential(std::vector<Node*>  &graph, int seedSetSize, int maxLevel, int seed, int n){
  std::vector<Node*> seedSet = generateSeedSet(seedSetSize, n, graph, seed);

  auto beg = std::chrono::high_resolution_clock::now();
  sequentialBFS(graph, seedSet, maxLevel);
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "Sequential BFS for " << n << " graph size with seed set size of "<< seedSetSize <<" : " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() 
              << " microseconds" << std::endl;
}

void processLevel(std::queue<Node*>& q, std::mutex& qMutex, std::atomic<bool>& done, std::vector<std::queue<Node*>> vq, std::queue<Node*>& threadQueue) {
    while (true) {
        Node* current = nullptr;
        {
            std::lock_guard<std::mutex> lock(qMutex);
            if (q.empty()) {
                done = true;
                return;
            }
            current = q.front();
            q.pop();
        }
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(10ms);
        if (calcNode(current)) {
            for (Node* child : current->children) {
                if (!child->markedBit) {
                    {
                        threadQueue.push(child); 
                    }
                    child->markedBit = true; //atomic
                }
            }
        }
    }
}

void multithreadedBFS(std::vector<Node*>& graph, std::vector<Node*> seedSet, int maxLevel, size_t numThreads, std::vector<std::thread>& threads) {
    std::vector<std::queue<Node*>> vq(maxLevel+1);
    std::mutex qMutex;

    for(Node* nd : seedSet){
      if(!nd->markedBit){
        vq[nd->level].push(nd);
        nd->markedBit = true;
      }
    }
    for(int level = 0; level<=maxLevel; level++){
      std::queue<Node*>& q = vq[level];
      if(vq[level].size()==0) continue;

      std::atomic<bool> done(false);
      std::vector<std::queue<Node*>> tq(numThreads);
      for (int i = 0; i < (int) numThreads; ++i) {
          threads[i]= std::thread(processLevel, std::ref(q), std::ref(qMutex), std::ref(done), std::ref(vq), std::ref(tq[i]));
      }

      
      for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

      while (!done.load()) {
          std::this_thread::yield();
      }
      for (int i = 0; i < (int) numThreads; ++i) {

        while(!tq[i].empty()){
            Node* top = tq[i].front();
            tq[i].pop();
            vq[top->level].push(top);
        }
      }
    }

}


void testingMultithreadedBFS(std::vector<Node*> &graph, int seedSetSize, int maxLevel, size_t numThreads,int seed, int n){
  std::vector<Node*> seedSet = generateSeedSet(seedSetSize, n, graph, seed);
  std::vector<std::thread> threads((int)numThreads);

  auto beg = std::chrono::high_resolution_clock::now();
  multithreadedBFS(graph, seedSet, maxLevel, numThreads, threads);
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "PoolScheduling MT BFS for " << n << " graph size with seed set size of "<< seedSetSize << " with thread count of "<< numThreads << " : " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() 
              << " microseconds" << std::endl;
}

int main() {
    int n = 10000;  // Number of nodes
    std::vector<Node*> graphSeq = generateAcyclicGraph(n, 42);
    //printGraph(graphSeq[0]);

    std::vector<Node*> graphMT1 = generateAcyclicGraph(n, 42);
    

    int maxLevelSeq = 0;
    for (Node* nd : graphSeq) {
        maxLevelSeq = std::max(nd->level, maxLevelSeq);
    }
    int maxLevelMT1 = 0;
    for (Node* nd : graphMT1) {
        maxLevelMT1 = std::max(nd->level, maxLevelMT1);
    }
    std::cout<<"Max level for the graph " <<maxLevelSeq<<std::endl;
    

    //int seed set size is 1000
    testingSequential(graphSeq, 1000, maxLevelSeq, 45, n);

    testingMultithreadedBFS(graphMT1, 1000, maxLevelMT1, 16,  48, n);
    
    // Clean up
    for (Node* node : graphSeq) {
        delete node;
    }

    for (Node* node : graphMT1) {
        delete node;
    }

    return 0;
}