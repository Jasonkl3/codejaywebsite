"use client";
import { useEffect, useState } from "react";

const ListCheckbox = ({ id }: { id: number }) => {
    const [checked, setChecked] = useState<boolean>(false);

    useEffect(() => {
        const storedCheckedItems = localStorage.getItem('checked-items');
        if (storedCheckedItems !== null) {
            const checkedItemsSet = new Set(JSON.parse(storedCheckedItems));
            setChecked(checkedItemsSet.has(id));
        }
    }, [id]);

    const handleChecked = () => {
        setChecked(prevChecked => {
            const newChecked = !prevChecked;
            const storedCheckedItems = localStorage.getItem('checked-items');
            let checkedItemsSet = storedCheckedItems ? new Set(JSON.parse(storedCheckedItems)) : new Set();

            if (newChecked) {
                checkedItemsSet.add(id);
            } else {
                checkedItemsSet.delete(id);
            }

            localStorage.setItem('checked-items', JSON.stringify(Array.from(checkedItemsSet)));
            return newChecked;
        });
    };
    
    return (
        <div className="flex items-center">
            <input 
                id="checkbox-all-search" 
                type="checkbox" 
                checked={checked} 
                onChange={handleChecked} 
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 dark:focus:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
            />
            <label htmlFor="checkbox-all-search" className="sr-only">checkbox</label>
        </div>
    );
};

export default ListCheckbox;