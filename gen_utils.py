import pandas as pd
import os

def load_dataset(base_dir='test_sets', task='all', source='all', language='all'):
    #test_sets/{task_name}/{source_name}/{language}.csv
    task = task.split(",")
    source = source.split(",")
    language = language.split(",")
    
    task_dirs = os.listdir(base_dir)
    
    dataset = []
    
    for task_dir in task_dirs:
        if 'all' not in task and task_dir not in task: continue
        
        source_dirs = os.listdir(os.path.join(base_dir, task_dir))
        for source_dir in source_dirs:
            if 'all' not in source and source_dir not in source: continue
            
            language_files = os.listdir(os.path.join(base_dir, task_dir, source_dir))
            for language_file in language_files:
                name, ext = os.path.splitext(os.path.basename(language_file))
                if 'all' not in language and name not in language: continue
                
                prompts = pd.read_csv(os.path.join(base_dir, task_dir, source_dir, language_file)).prompt.tolist()
                
                for prompt in prompts:
                    dataset.append({"prompt":prompt, "task":task_dir, "source":source_dir, "language":name})
    
    return dataset