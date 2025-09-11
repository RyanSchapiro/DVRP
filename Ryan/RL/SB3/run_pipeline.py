import os
import sys
from datetime import datetime

sys.path.append('.')

def run():

    num_instances = int(input("Training instances [10]: ").strip() or "10")
    total_steps = int(input("Total steps [100000]: ").strip() or "100000")
    vehicle_cost = float(input("Vehicle cost [50]: ").strip() or "50")
    max_customers = int(input("Max customers [100]: ").strip() or "100")
    test_file = input("Test file [c1_25.txt]: ").strip() or "c1_25.txt"
    test_episodes = int(input("Test episodes [20]: ").strip() or "20")
    
    
    model_name = f"{max_customers}_{max_customers}"

    try:

        
        from td import train_model
        model = train_model(
            total_steps=total_steps,
            vehicle_cost=vehicle_cost,
            max_customers=max_customers
        )
        
        # Save model
        print(f"Saving model: {model_name}")
        model.save(model_name)

        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            return
        
        from test import evaluate_model, analyze, save
    
        print(f"Testing on: {test_file}")
        results = evaluate_model(
            model, test_file,
            num_episodes=test_episodes,
            vehicle_cost=vehicle_cost,
            max_customers=max_customers
        )
        
        # Analyze and save results
        results = analyze(results, vehicle_cost, test_file)
        results_file = save(results, vehicle_cost, test_file, model_name)
        
        print(f"  Model: {model_name}.zip")
        print(f"  Results: {results_file}")
        
        if results:
            objectives = [r['objective_cost'] for r in results]
            vehicles = [r['vehicles'] for r in results]
            print(f"  Complete solutions: {len(results)}/{len(results)}")
            print(f"  Best objective: {min(objectives):.2f}")
            print(f"  Avg objective: {sum(objectives)/len(objectives):.2f}")
            print(f"  Avg vehicles: {sum(vehicles)/len(vehicles):.1f}")
        else:
            print(f"  ❌ No complete solutions found")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all files are in the same directory")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


def train():

    try:
        from td import main
        main()
    except ImportError:
        print("❌ td.py not found")
    except Exception as e:
        print(f"❌ Training failed: {e}")


def test():
    
    try:
        from test import main
        main()
    except ImportError:
        print("❌ test.py not found")
    except Exception as e:
        print(f"❌ Testing failed: {e}")


def main():

    print("1. Run project")
    print("2. Training only")
    print("3. Testing only")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        run()
    elif choice == "2":
        train()
    elif choice == "3":
        test()


if __name__ == "__main__":
    main()