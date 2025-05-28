import sys
import pandas as pd
import numpy as np
import base64
import io
from PIL import Image
import gradio as gr
from rdkit import Chem
from rdkit.Chem import Draw

# Import the function from run.py and other required components
from run import get_final_smiles, load_digress_config
from filterer import SMILESFilterer

def load_generator():
    # Import DiGress generator
    sys.path.append("./generators/DiGress/")
    sys.path.append("./generators/DiGress/src/")
    digress_cfg = load_digress_config()
    from generator import DigressGenerator
    generator = DigressGenerator(digress_cfg)
    return generator

def numpy_to_base64(img_array):
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def process(num_molecules, lower_logP, upper_logP, lower_SA, upper_SA, lower_pIC50, upper_pIC50, max_rings_count):
    # Initialize generator and filterer
    generator = load_generator()
    filterer = SMILESFilterer()
    
    thresholds = [lower_logP, upper_logP, lower_SA, upper_SA, lower_pIC50, upper_pIC50, max_rings_count]
    # Generate SMILES using get_final_smiles function
    generated_smiles = get_final_smiles(generator, filterer, thresholds, num_molecules, scale_up=20)
    
    # Filter by property ranges as needed
    results = []
    for result in generated_smiles:
        smiles, logP, SA, pic50 = result
            
        # Visualize molecule
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol)
        img_array = np.array(img)
        img_base64 = numpy_to_base64(img_array)
        
        results.append((smiles, logP, SA, pic50, img_base64))
    
    # Generate HTML table
    table_html = """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: center;
        }
        td {
            word-wrap: break-word;
            white-space: normal;
        }
        .col-name {
            width: 20%;
        }
        .col-idx {
            width: 5%;
        }
        .col-scores {
            width: 10%;
        }
        .col-image {
            width: 50%;
        }
        img {
            max-width: 100%;
            height: auto;
        }
    </style>
    """
    table_html += "<table>"
    table_html += "<tr><th class='col-idx'>Index</th><th class='col-name'>SMILES</th><th class='col-scores'>logP</th><th class='col-scores'>SA</th><th class='col-scores'>pIC50</th><th class='col-image'>Visualization</th></tr>"
    
    for i, (smiles, logP, SA, pic50, img) in enumerate(results, 1):
        table_html += f"<tr><td class='col-idx'>{i}</td><td class='col-name'>{smiles}</td><td class='col-scores'>{logP:.2f}</td><td class='col-scores'>{SA:.2f}</td><td class='col-scores'>{pic50:.2f}</td><td class='col-image'><img src='{img}'></td></tr>"
    
    table_html += "</table>"
    
    # Store data for export
    global generated_data
    generated_data = results
    
    return table_html

def export_to_csv():
    global generated_data
    if not generated_data:
        return "No data to export", None

    df = pd.DataFrame({
        "SMILES": [mol[0] for mol in generated_data],
        "logP": [mol[1] for mol in generated_data],
        "SA": [mol[2] for mol in generated_data],
        "pIC50": [mol[3] for mol in generated_data]
    })
    # Save to CSV
    csv_file = "generated_molecules.csv"
    df.to_csv(csv_file, index=False)
    return "CSV file is ready for download", csv_file

# Initialize global variable for generated data
generated_data = []

# Create Gradio interface
def main():
    print("Starting Gradio interface...")
    with gr.Blocks() as demo:
        gr.Markdown("# Molecule Generator: DiGress")
        gr.Markdown("# pIC50 Predictor: GNN")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### General Settings")
                n_sample_to_generate = gr.Number(label="Number of molecules to generate", value=10, precision=0)
                max_rings_count = gr.Number(label="Maximum number of large rings allowed", value=2)
            
            with gr.Column():
                gr.Markdown("### LogP Range")
                lower_logP = gr.Number(label="Min", value=1)
                upper_logP = gr.Number(label="Max", value=4)
            
            with gr.Column():
                gr.Markdown("### Synthetic Accessibility Range")
                lower_SA = gr.Number(label="Min", value=1)
                upper_SA = gr.Number(label="Max", value=3)
                
            with gr.Column():
                gr.Markdown("### pIC50 Range")
                lower_pIC50 = gr.Number(label="Min", value=8)
                upper_pIC50 = gr.Number(label="Max", value=12)
    
        generate_button = gr.Button("Generate Molecules")
        export_button = gr.Button("Export to CSV")
    
        export_message = gr.Textbox(label="Export Status", interactive=False, visible=False)
        export_file = gr.File(label="Download CSV", visible=False)
        output = gr.HTML(label="Generated Molecules")
    
        generate_button.click(
            process,
            inputs=[n_sample_to_generate, lower_logP, upper_logP, lower_SA, upper_SA, lower_pIC50, upper_pIC50, max_rings_count],
            outputs=output
        )

        export_button.click(
            export_to_csv,
            inputs=[],
            outputs=[export_message, export_file]
        )

        export_button.click(
            lambda: [gr.update(visible=True), gr.update(visible=True)],
            inputs=[],
            outputs=[export_message, export_file]
        )

    demo.launch(share=True)
    
if __name__ == '__main__':
    main()


