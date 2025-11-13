import argparse
import torch
import tiktoken
from src.config import GEMMA3_CONFIG_270M
from src.model import Gemma3Model

def run_inference(sentence, max_new_tokens=200, temperature=1.0, top_k=None, model_path="best_model_params.pt"):
    enc = tiktoken.get_encoding("gpt2")
    model = Gemma3Model(GEMMA3_CONFIG_270M)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model = model.to(device)
    context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(0).to(device)
    output = model.generate(context, max_new_tokens, temperature, top_k)
    return enc.decode(output.squeeze().tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the trained Gemma3 model.")
    parser.add_argument("sentence", type=str, nargs="?", default="Once upon a time there was a pumpkin.",
                        help="Input sentence for generation (default: 'Once upon a time there was a pumpkin.')")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Maximum number of new tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k sampling value (default: None)")
    parser.add_argument("--model_path", type=str, default="best_model_params.pt",
                        help="Path to the trained model parameters (default: 'best_model_params.pt')")
    
    args = parser.parse_args()
    
    print(f"Input: {args.sentence}")
    output = run_inference(
        args.sentence,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.model_path
    )
    print(f"Output: {output}")