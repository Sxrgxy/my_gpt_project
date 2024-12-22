from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Load the model and tokenizer
    model_name = "gpt2" #"microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Welcome to gpt2 chat! Type 'exit' to quit.")

    #Dialogue loop
    while True:
        # Get user input
        prompt = input("\nYou: ")
        if prompt.lower() == "exit":
            print("Goodbye!")
            break
        # Prepare inputs for the model
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate response
        outputs = model.generate(
            **inputs, 
            max_length=100, 
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display the response
        print("gpt2:", response)

if __name__ == "__main__":
    main()
