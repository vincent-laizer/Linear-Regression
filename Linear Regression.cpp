
#include "Loader.h"

int main()
{
    /*univariant linear regression*/
    float input, prediction;
    string filename; //file containing training data set

    cout << "Enter dataset filename: ";
    cin >> filename;

    gradientDescent(filename);

    while(true) {
        cout << "Enter Value to Predict: ";
        cin >> input;
        prediction = predict(input, 10000);
        cout << "Predicted Value is: " << prediction << endl;
    }

    return 0;
}