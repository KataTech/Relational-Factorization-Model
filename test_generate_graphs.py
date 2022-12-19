from dev import generate_util as gen

GENERATIVE_FUNCS = [gen.gen_cycle_graph, gen.gen_star_graph]
MAX_NODE = 35
NUM_GRAPHS = 10
PERMUTE = False 


# TODO: Uncomment the following for testing the two generative functions 
# print(gen.gen_cycle_graph(25))
# print(gen.gen_star_graph(25, verbose = True))

# TODO: Uncomment the following for testing random_generate()
# graph_dict = gen.random_generate(GENERATIVE_FUNCS, MAX_NODE, NUM_GRAPHS, PERMUTE)
# for key, val in graph_dict.items(): 
#     print(f"Class {key} ---------------------------- \n")
#     print(f"    Number of elements: {len(val)}")
#     print(f"    Printing first element....")
#     print(val[0])
#     print(f"    Printing second element....")
#     print(val[1])
#     print("----------------------------------------- \n")

# Generate star and cycle graphs 
graph_dict = gen.random_generate(GENERATIVE_FUNCS, MAX_NODE, NUM_GRAPHS, PERMUTE)
gen.graph2text(graph_dict, "Cycles_and_Stars", del_existing = True)


