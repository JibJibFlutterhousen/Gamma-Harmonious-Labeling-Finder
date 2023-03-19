import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import os
import time

def _make_graph_directory(graph):
    graph_name = _graph6_bytes_to_file_name(nx.to_graph6_bytes(graph,header=False).strip())
    folders_to_make = ["Graphs", f"Graphs/{graph_name}"]
    [os.mkdir(folder_name) for folder_name in folders_to_make if not os.path.exists(folder_name)]
    _draw_graph(graph, f"Graphs/{graph_name}/Graph {graph_name}")
    return


def _get_labeling_set(mods):
    labels = list([(item,) for item in range(mods[0])])
    for index,mod in enumerate(mods[1:]):
        for item in [label for label in labels if len(label) == index+1]:
            [labels.append(item+(new_item,)) for new_item in range(mod)]
    remove_items = [item for item in labels if len(item) < len(mods)]
    [labels.remove(item) for item in remove_items]
    return tuple(labels)

def _export_labeled_graph(graph, labeling, mods, path=None):
    if path==None:
        _draw_labeled_graph(graph, labeling, mods, path)
    else:
        _draw_labeled_graph(graph, labeling, mods, path)
        _write_labeled_graph(graph, labeling, mods, path)
    return

def _draw_graph(graph, path=None):
    figure,axes = plt.subplots(figsize=(10,10), dpi=250)
    axes.set_title(f"{_graph6_bytes_to_file_name(nx.to_graph6_bytes(graph,header=False).strip())}")
    nx.draw_kamada_kawai(graph, with_labels=True, edge_color="black", node_color="grey", font_color="black")
    plt.tight_layout()
    if path == None:
        plt.show()
    else:
        plt.savefig(f"{path}.png")
        plt.close()
    return

def _draw_labeled_graph(graph, labeling, mods, path=None):
    figure,axes = plt.subplots(figsize=(10,10), dpi=250)
    axes.set_title(f"Labeling found with {[f'Z_{mod}' for mod in mods]}")
    position = nx.kamada_kawai_layout(graph)
    nx.draw(graph, position, node_color="grey", edge_color="black")
    nx.draw_networkx_labels(graph, position, labels=_get_node_label_mapping(graph,labeling), font_color="black")
    nx.draw_networkx_edge_labels(graph, position, _get_edge_label_mapping(graph,labeling,mods), font_color="black")
    plt.tight_layout()
    if path == None:
        plt.show()
    else:
        plt.savefig(f"{path}.png")
        plt.close()
    return

def _write_labeled_graph(graph, labeling, mods, path):
    with open(f"{path}.txt", "w") as out_file:
        [out_file.write(f"{key}:{value}\n") for key,value in _get_node_label_mapping(graph,labeling).items()]
    return

def _graph6_bytes_to_file_name(graph6_bytes):
    """
        Ok, so hold onto your butts for this one, because windows caused a bug here.
        First, you cannot use the symbols "?", "\", or "|" in a windows file name but they can be encoded as graph6 byte strings... so we replace them with "1", "2", and "3" respectively.
        Next, windows cannot have files with the same letters, but different cases, in the same directory (i.e. "Bw.g6" and "BW.g6") but this CAN happen for non-isomorphic graphs. So we choose to make everything lowercase.
        If an upper-case letter is replaced with a lower-case by this function, it will append a "+" symbol in front so it can be decoded later.
    """
    return graph6_bytes.decode().strip().replace("?","1").replace(chr(92),"2").replace("|","3").replace('A','+a').replace('B','+b').replace('C','+c').replace('D','+d').replace('E','+e').replace('F','+f').replace('G','+g').replace('H','+h').replace('I','+i').replace('J','+j').replace('K','+k').replace('L','+l').replace('M','+m').replace('N','+n').replace('O','+o').replace('P','+p').replace('Q','+q').replace('R','+r').replace('S','+s').replace('T','+t').replace('U','+u').replace('V','+v').replace('W','+w').replace('X','+x').replace('Y','+y').replace('Z','+z')+".g6"


def _allocate_work(iterator, worker_id, num_workers):
    for job_number,job in enumerate(iterator):
        if (job_number % num_workers) == worker_id:
            yield job
        else:
            continue

def _get_node_label_mapping(graph, new_labeling):
    labeling_iterator = iter(new_labeling)
    mapping = dict()
    [mapping.update({vertex:next(labeling_iterator)}) for vertex in graph.nodes()]
    return mapping

def _combine(item1, item2, mods):
    item3 = []
    for index,mod in enumerate(mods):
        item3.append((item1[index]+item2[index])%mod)
    return tuple(item3)

def _get_edge_label_mapping(graph, new_labeling, mods):
    labeling_function = _get_node_label_mapping(graph, new_labeling)
    mapping = dict()
    [mapping.update({edge:_combine(labeling_function[edge[0]],labeling_function[edge[1]],mods)}) for edge in graph.edges()]
    return mapping

def _is_valid_labeling(graph, labeling_set, mods):
    if len(set(_get_edge_label_mapping(graph, labeling_set, mods).values())) == graph.number_of_edges():
        return True
    else:
        return False

def get_labeling(graph, labels, mods):
    print(f"Getting a single labeling for the graph {_graph6_bytes_to_file_name(nx.to_graph6_bytes(graph, header=False).strip())}")
    if len(labels) != graph.number_of_edges():
        print(f"The supplied list of labels does not match the number of edges in the supplied graph")
        raise NotImplementedError
    _make_graph_directory(graph)
    if not nx.is_tree(graph):
        _get_labeling_non_tree(graph, labels, mods)
    else:
        _get_labeling_tree(graph, labels, mods)
    return

def _get_labeling_non_tree(graph, labels, mods):
    num_workers = max(1, multiprocessing.cpu_count()-1)
    workers = []
    for worker_id in range(num_workers):
        job = multiprocessing.Process(target=_get_labeling_non_tree_helper, args=(graph, labels, mods, worker_id, num_workers))
        job.start()
        workers.append(job)
    output = _get_labeling_non_tree_waiter(graph, labels, num_workers)
    if output == None:
        print(f"No valid labeling found")
        with open(f"Graphs/{_graph6_bytes_to_file_name(nx.to_graph6_bytes(graph, header=False).strip())}/Output.txt", "w") as out_file:
            out_file.write(f"No labeling of {_graph6_bytes_to_file_name(nx.to_graph6_bytes(graph, header=False).strip())} is found with {labels}\n")
    else:
        print(f"Valid labeling found!\n{_get_node_label_mapping(graph,output)}")
        path = f"Graphs/{_graph6_bytes_to_file_name(nx.to_graph6_bytes(graph, header=False).strip())}/Valid_Gamma_Harmonious_Labeling"
        _export_labeled_graph(graph, labels, mods, path)
    return

def _get_labeling_tree(graph, labels, mods):
    num_workers = max(1, multiprocessing.cpu_count()-1)
    workers = []
    for worker_id in range(num_workers):
        job = multiprocessing.Process(target=_get_labeling_tree_helper, args=(graph, labels, mods, worker_id, num_workers))
        job.start()
        workers.append(job)
    output = _get_labeling_tree_waiter(graph, labels, num_workers)
    if output == None:
        print(f"No valid labeling found")
        with open(f"Graphs/{_graph6_bytes_to_file_name(nx.to_graph6_bytes(graph, header=False).strip())}/Output.txt", "w") as out_file:
            out_file.write(f"No labeling of {_graph6_bytes_to_file_name(nx.to_graph6_bytes(graph, header=False).strip())} is found with {labels}\n")
    else:
        print(f"Valid labeling found!\n{_get_node_label_mapping(graph,output)}")
        path = f"Graphs/{_graph6_bytes_to_file_name(nx.to_graph6_bytes(graph, header=False).strip())}/Valid_Gamma_Harmonious_Labeling"
        _export_labeled_graph(graph, labels, mods, path)
    return

def _get_labeling_non_tree_helper(graph, labels, mods, worker_id, num_workers):
    for combination in _allocate_work(it.permutations(labels, graph.number_of_nodes()), worker_id, num_workers):
        if _is_valid_labeling(graph, combination, mods):
            with open(f"valid_labeling_worker_{worker_id}.pickle", "wb") as out_file:
                pickle.dump(combination, out_file)
            break
    else:
        with open(f"no_labeling_found_with_worker_{worker_id}.txt", "w") as out_file:
            out_file.write("")
    return

def _get_labeling_tree_helper(graph, labels, mods, worker_id, num_workers):
    for combination in _allocate_work(it.permutations(labels, graph.number_of_nodes()-1), worker_id, num_workers):
        for repeated_label in labels:
            new_combination = combination + (repeated_label,)
            if _is_valid_labeling(graph, new_combination, mods):
                with open(f"valid_labeling_worker_{worker_id}.pickle", "wb") as out_file:
                    pickle.dump(new_combination, out_file)
                break
        else:
            continue
        break
    else:
        with open(f"no_labeling_found_with_worker_{worker_id}.txt", "w") as out_file:
            out_file.write("")  
    return

def _get_labeling_non_tree_waiter(graph, labels, num_workers):
    done = False
    output = None
    while not done:
        time.sleep(2)
        valid_labeling_files = [file_name for file_name in os.listdir() if "valid_labeling_worker_" in file_name]
        exhausted_worker_files = [file_name for file_name in os.listdir() if "no_labeling_found_with_worker_" in file_name]
        if len(exhausted_worker_files) == num_workers:
            done = True
        for file in valid_labeling_files:
            if os.path.getsize(file) > 0:
                with open(file, "rb") as in_file:
                    output = pickle.load(in_file)
                [child.terminate() for child in multiprocessing.active_children()]
                done = True
    for file in valid_labeling_files:
        os.remove(file)
    for file in exhausted_worker_files:
        os.remove(file)
    return output

def _get_labeling_tree_waiter(graph, labels, num_workers):
    done = False
    output = None
    while not done:
        time.sleep(2)
        valid_labeling_files = [file_name for file_name in os.listdir() if "valid_labeling_worker_" in file_name]
        exhausted_worker_files = [file_name for file_name in os.listdir() if "no_labeling_found_with_worker_" in file_name]
        if len(exhausted_worker_files) == num_workers:
            done = True
        for file in valid_labeling_files:
            if os.path.getsize(file) > 0:
                with open(file, "rb") as in_file:
                    output = pickle.load(in_file)
                [child.terminate() for child in multiprocessing.active_children()]
                done = True
    for file in valid_labeling_files:
        os.remove(file)
    for file in exhausted_worker_files:
        os.remove(file)
    return output

def get_all_labelings(graph, labels, mods):
    print(f"Getting a all labelings for the graph {_graph6_bytes_to_file_name(nx.to_graph6_bytes(graph, header=False).strip())}")
    if len(labels) != graph.number_of_edges():
        print(f"The supplied list of labels does not match the number of edges in the supplied graph")
        raise NotImplementedError
    _make_graph_directory(graph)
    if not nx.is_tree(graph):
        _get_all_labeling_non_tree(graph, labels, mods)
    else:
        _get_all_labeling_tree(graph, labels, mods)
    return

def _get_all_labeling_non_tree(graph, labels, mods):
    num_workers = max(1, multiprocessing.cpu_count()-1)
    workers = []
    for worker_id in range(num_workers):
        job = multiprocessing.Process(target=_get_all_labeling_non_tree_helper, args=(graph, labels, mods, worker_id, num_workers))
        job.start()
        workers.append(job)
    for worker in workers:
        worker.join()
    return

def _get_all_labeling_tree(graph, labels, mods):
    num_workers = max(1, multiprocessing.cpu_count()-1)
    workers = []
    for worker_id in range(num_workers):
        job = multiprocessing.Process(target=_get_all_labeling_tree_helper, args=(graph, labels, mods, worker_id, num_workers))
        job.start()
        workers.append(job)
    for worker in workers:
        worker.join()
    return

def _get_all_labeling_non_tree_helper(graph, labels, mods, worker_id, num_workers):
    valid_labelings = 0
    path = f"Graphs/{_graph6_bytes_to_file_name(nx.to_graph6_bytes(graph, header=False).strip())}/Worker-{worker_id}_Labeling"
    for combination in _allocate_work(it.permutations(labels, graph.number_of_nodes()), worker_id, num_workers):
        if _is_valid_labeling(graph, combination, mods):
            valid_labelings += 1
            new_path = f"{path}-{valid_labelings}"
            _export_labeled_graph(graph, combination, mods, new_path)

def _get_all_labeling_tree_helper(graph, labels, mods, worker_id, num_workers):
    valid_labelings = 0
    path = f"Graphs/{_graph6_bytes_to_file_name(nx.to_graph6_bytes(graph, header=False).strip())}/Worker-{worker_id}_Labeling"
    for combination in _allocate_work(it.permutations(labels, graph.number_of_nodes()-1), worker_id, num_workers):
        for repeated_label in labels:
            new_combination = combination + (repeated_label,)
            if _is_valid_labeling(graph, new_combination, mods):
                valid_labelings += 1
                new_path = f"{path}-{valid_labelings}"
                _export_labeled_graph(graph, new_combination, mods, new_path)